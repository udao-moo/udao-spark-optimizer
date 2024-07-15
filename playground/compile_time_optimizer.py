import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
import torch as th

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_compile,
)
from udao_spark.optimizer.setup import q_compile_setup
from udao_spark.optimizer.utils import weighted_utopia_nearest
from udao_spark.utils.logging import logger
from udao_spark.utils.monitor import UdaoMonitor
from udao_spark.utils.params import get_ag_parameters
from udao_trace.utils import JsonHandler


def get_params() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--ag_model_q_latency", type=str, default=None,
                        help="specific model name for AG for latency",)
    parser.add_argument("--ag_model_q_io", type=str, default=None,
                        help="specific model name for AG for IO",)
    parser.add_argument("--n_conf_samples", type=int, default=10000,
                        help="number of configurations to sample")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode")
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable ensemble mode")
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_params().parse_args()
    base_dir = Path(__file__).parent

    meta = q_compile_setup(base_dir, params)
    atomic_optimizer = meta["atomic_optimizer"]
    templates = meta["templates"]
    raw_traces = meta["raw_traces"]
    ag_model = meta["ag_model"]

    device = "gpu" if th.cuda.is_available() else "cpu"
    use_ag = params.ensemble
    graph_choice = params.graph_choice
    bm = params.benchmark
    wun_weights_pairs = {
        0: (0.0, 1.0),
        1: (0.1, 0.9),
        5: (0.5, 0.5),
        9: (0.9, 0.1),
        10: (1.0, 0.0),
    }

    target_confs: Dict[str, Dict] = {}
    total_monitor = {}
    target_objs: Dict[str, Dict] = {}
    n_samples = params.n_conf_samples
    for template, trace in zip(templates, raw_traces):
        logger.info(f"Processing {trace}")
        query_id = f"{template}-1"
        non_decision_input = get_non_decision_inputs_for_q_compile(trace)
        monitor = UdaoMonitor()
        po_objs, po_confs, _ = atomic_optimizer.solve(
            template=template,
            non_decision_input=non_decision_input,
            seed=params.seed,
            use_ag=use_ag,
            ag_model=ag_model,
            sample_mode="lhs",
            n_samples=n_samples,
            monitor=monitor,
            ercilla=True if device == "gpu" else False,
            graph_choice=graph_choice,
        )
        total_monitor[query_id] = monitor.to_dict()
        if po_objs is None or po_confs is None:
            logger.warning(f"Failed to solve {template}")
            continue
        target_confs[query_id] = {}
        target_objs[query_id] = {}
        for k, wun_weights in wun_weights_pairs.items():
            reco_obj, reco_conf = weighted_utopia_nearest(
                pareto_objs=po_objs,
                pareto_confs=po_confs,
                weights=np.array(wun_weights),
            )
            target_confs[query_id][k] = ",".join(reco_conf)
            target_objs[query_id][k] = {
                "latency_s_hat": float(reco_obj[0]),
                "cost_hat": float(reco_obj[1]),
            }

    todo_confs = {
        query_id: np.unique([c for c in confs_dict.values()]).tolist()
        for query_id, confs_dict in target_confs.items()
    }

    if not use_ag:
        suffix = f"{n_samples}_{graph_choice}_{device}"
    else:
        ag_model_short = "_".join(f"{k.split('_')[0]}:{v}" for k, v in ag_model.items())
        suffix = f"{n_samples}_{graph_choice}_em({ag_model_short})_{device}"

    torun_file = f"compile_time_output/{bm}100/lhs/moo-run_confs_{suffix}.json"
    toanalyze_file = f"compile_time_output/{bm}100/lhs/moo-full_confs_{suffix}.json"
    runtime_file = f"compile_time_output/{bm}100/lhs/moo-runtime_{suffix}.json"
    target_objs_file = f"compile_time_output/{bm}100/lhs/moo-objs_{suffix}.json"

    os.makedirs(os.path.dirname(torun_file), exist_ok=True)
    JsonHandler.dump_to_file(target_confs, toanalyze_file, indent=2)
    JsonHandler.dump_to_file(todo_confs, torun_file, indent=2)
    JsonHandler.dump_to_file(total_monitor, runtime_file, indent=2)
    JsonHandler.dump_to_file(target_objs, target_objs_file, indent=2)
