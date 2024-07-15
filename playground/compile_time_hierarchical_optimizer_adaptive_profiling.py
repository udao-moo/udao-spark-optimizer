import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from udao.optimization.utils.moo_utils import is_pareto_efficient

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_qs_compile_dict,
)
from udao_spark.optimizer.hierarchical_optimizer import HierarchicalOptimizer
from udao_spark.optimizer.utils import get_ag_meta, weighted_utopia_nearest
from udao_spark.utils.params import QType, get_compile_time_optimizer_parameters
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, JsonHandler, PickleHandler
from udao_trace.utils.logging import logger
from udao_trace.workload import Benchmark

logger.setLevel("INFO")


def get_params() -> argparse.ArgumentParser:
    parser = get_compile_time_optimizer_parameters()
    parser.add_argument("--target_templates", nargs="+", type=str, default=None)
    parser.add_argument("--ordered_queries_header", type=str, default="assets")
    return parser


if __name__ == "__main__":
    # Initialize InjectionExtractor
    params = get_params().parse_args()
    logger.info(f"get parameters: {params}")
    if params.sample_mode.startswith("grid") and params.selected_features:
        raise ValueError("grid search does not support selected features")
    if not params.verbose:
        logger.setLevel("ERROR")

    bm = params.benchmark
    q_type: QType = params.q_type
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size

    if q_type not in ["qs_lqp_compile", "qs_lqp_runtime"]:
        raise ValueError(
            f"q_type {q_type} is not supported for "
            f"compile time hierarchical optimizer"
        )

    ag_meta = get_ag_meta(
        bm,
        hp_choice,
        graph_choice,
        q_type,
        ag_sign,
        infer_limit,
        infer_limit_batch_size,
        params.ag_time_limit,
        params.fold,
    )

    # Initialize HierarchicalOptimizer and Model
    base_dir = Path(__file__).parent
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"]
    )

    hier_optimizer = HierarchicalOptimizer(
        bm=bm,
        model_sign=ag_meta["model_sign"],
        graph_model_params_path=ag_meta["model_params_path"],
        graph_weights_path=ag_meta["graph_weights_path"],
        q_type=params.q_type,
        data_processor_path=ag_meta["data_processor_path"],
        spark_conf=spark_conf,
        decision_variables=decision_variables,
        ag_path=ag_meta["ag_path"],
        clf_json_path=None
        if params.disable_failure_clf
        else str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
        clf_recall_xhold=params.clf_recall_xhold,
    )

    # Prepare traces
    path = (
        f"{params.ordered_queries_header}/{bm}_physical_query_plans_ordered_paths.json"
    )
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} does not exist")
    ordered_traces_dict = JsonHandler.load_json(path)
    target_templates = params.target_templates or Benchmark(BenchmarkType(bm)).templates
    target_queries = [f"{t}-1" for t in target_templates]
    target_traces = {}
    for query_id, ordered_traces in ordered_traces_dict.items():
        for trace in ordered_traces:
            if not Path(trace).exists():
                print(f"{trace} does not exist")
                raise FileNotFoundError(f"{trace} does not exist")
        if query_id in target_queries:
            target_traces[query_id] = ordered_traces
            print(f"Found {query_id} with {len(ordered_traces)} traces")

    # Compile time QS logical plans from CBO estimation (a list of LQP-sub)
    is_oracle = q_type == "qs_lqp_runtime"
    use_ag = not params.use_mlp

    if params.moo_algo == "evo":
        param_meta = {
            "pop_size": params.pop_size,
            "nfe": params.nfe,
        }
    elif params.moo_algo == "ws":
        param_meta = {
            "n_samples": params.n_samples,
            "n_ws": params.n_ws,
        }
    elif "hmooc" or "div_and_conq_moo" in params.moo_algo:
        param_meta = {
            "n_c_samples": params.n_c_samples,
            "n_p_samples": params.n_p_samples,
        }
    elif params.moo_algo == "ppf":
        param_meta = {
            "n_process": params.n_process,
            "n_grids": params.n_grids,
            "n_max_iters": params.n_max_iters,
        }
    elif params.moo_algo == "analyze_model_accuracy" or params.moo_algo == "test":
        param_meta = {}
    else:
        raise Exception(f"algo {params.moo_algo} is not supported!")

    ag_model = {
        "ana_latency_s": params.ag_model_qs_ana_latency or "WeightedEnsemble_L2_FULL",
        "io_mb": params.ag_model_qs_io or "CatBoost",
    }

    selected_features: Optional[Dict[str, List[str]]] = (
        {
            "c": [f"k{i}" for i in [1, 2, 3, 4, 6, 7]],
            "p": [f"s{i}" for i in [1, 4, 5, 8, 9]],
            "s": [],
        }
        if params.selected_features
        else None
    )

    po_set_dict = defaultdict(list)
    solving_time_dict = defaultdict(list)

    for query_id, raw_traces in target_traces.items():
        logger.info(f"Processing {query_id} with {len(raw_traces)} traces")

        current_po_objs = None
        current_po_confs = None

        for trace_id, trace in enumerate(raw_traces):
            non_decision_input = get_non_decision_inputs_for_qs_compile_dict(
                trace, is_oracle=is_oracle
            )
            po_objs, po_conf, dt = hier_optimizer.solve(
                template=query_id.split("-")[0],
                non_decision_input=non_decision_input,
                seed=params.seed,
                use_ag=use_ag,
                ag_model=ag_model,
                algo=params.moo_algo,
                save_data=params.save_data,
                query_id=query_id,
                sample_mode=params.sample_mode,
                param_meta=param_meta,
                time_limit=params.time_limit,
                is_oracle=is_oracle,
                save_data_header=params.save_data_header,
                is_query_control=params.set_query_control,
                benchmark=bm,
                selected_features=selected_features,
            )
            solving_time_dict[query_id].append(dt)
            if po_objs is None or po_conf is None:
                logger.warning(f"Failed to solve {query_id}")
                continue

            if current_po_objs is None:
                current_po_objs = po_objs
                current_po_confs = po_conf
            else:
                new_objs_ids = [
                    po_id
                    for po_id, po_obj in enumerate(po_objs)
                    if not np.any(np.all(current_po_objs == po_obj, axis=1))
                ]
                # only append new objectives to avoid po_conf inconsistency
                if len(new_objs_ids) > 0:
                    current_po_objs = np.vstack(
                        [current_po_objs, po_objs[new_objs_ids]]
                    )
                    current_po_confs = np.hstack(
                        [current_po_confs, po_conf[new_objs_ids]]
                    )
                    po_inds = is_pareto_efficient(current_po_objs)
                    current_po_objs = current_po_objs[po_inds]
                    current_po_confs = current_po_confs[po_inds]

            current_conf_strs = np.array(
                [
                    ",".join([dict(conf)[k] for k in spark_conf.knob_names])
                    for conf in current_po_confs
                ]
            )
            po_set_dict[query_id].append(
                {
                    "po_objs": current_po_objs,
                    "po_confs": current_conf_strs,
                }
            )

            print(
                f"current_po_objs by consuming {trace_id} "
                f"QS structures: {len(current_po_objs)}"
            )

    # print(torun_json)
    name_prefix = "selected_params_" if selected_features is not None else ""
    fname_prefix = (
        f"adaptive_{name_prefix}nc{params.n_c_samples}_np{params.n_p_samples}"
    )
    fname_suffix = (
        "all" if params.target_templates is None else "+".join(params.target_templates)
    )

    for weights in [[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.0, 1.0]]:
        fname = (
            f"{fname_prefix}_{'_'.join([f'{w:.1f}' for w in weights])}_{fname_suffix}"
        )
        torun_json = {
            query_id: [
                weighted_utopia_nearest(
                    po_set["po_objs"], po_set["po_confs"], np.array(weights)
                )[1]
                for po_set in po_set_list
            ]
            for query_id, po_set_list in po_set_dict.items()
        }
        torun_3plan_json = {
            query_id: [conf_list[-1] if len(conf_list) < 3 else conf_list[2]]
            for query_id, conf_list in torun_json.items()
        }

        torun_json_distinct = {
            query_id: list(set(conf_strs)) for query_id, conf_strs in torun_json.items()
        }

        JsonHandler.dump_to_file(
            torun_json,
            file=f"{params.conf_save}/{bm}100/{params.moo_algo}_{params.sample_mode}/{fname}.json",
            indent=2,
        )
        JsonHandler.dump_to_file(
            torun_3plan_json,
            file=f"{params.conf_save}/{bm}100/{params.moo_algo}_{params.sample_mode}/{fname}_3plans.json",
            indent=2,
        )
        JsonHandler.dump_to_file(
            torun_json_distinct,
            file=f"{params.conf_save}/{bm}100/{params.moo_algo}_{params.sample_mode}/{fname}_distinct.json",
            indent=2,
        )

        pref = int(10 * weights[0])
        pref_dict = {
            pref: pd.DataFrame.from_dict(torun_3plan_json, orient="index")
            .reset_index()
            .rename(columns={"index": "query_id", 0: "conf"})
        }
        pred_dict_file = (
            f"pref_to_df_{params.moo_algo}_{params.sample_mode}_{fname}_3plans.pkl"
        )
        PickleHandler.save(
            pref_dict, f"{params.conf_save}/{bm}100/", pred_dict_file, overwrite=True
        )

    JsonHandler.dump_to_file(
        solving_time_dict,
        file=f"{params.conf_save}/{bm}100/{params.moo_algo}_{params.sample_mode}/{fname_prefix}_{fname_suffix}_solving_time.json",
        indent=2,
    )
    PickleHandler.save(
        po_set_dict,
        header=f"{params.conf_save}/{bm}100/{params.moo_algo}_{params.sample_mode}",
        file_name=f"{fname_prefix}_{fname_suffix}_po_set_dict.pkl",
        overwrite=True,
    )
    logger.info("Done!")
