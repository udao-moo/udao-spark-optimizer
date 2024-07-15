import copy
import itertools
import signal
import time
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from udao.optimization.concepts import (
    BoolVariable,
    FloatVariable,
    IntegerVariable,
    Objective,
    Variable,
)
from udao.optimization.concepts.problem import MOProblem
from udao.optimization.moo.progressive_frontier import ParallelProgressiveFrontier
from udao.optimization.soo.random_sampler_solver import RandomSamplerSolver

from udao_spark.optimizer.base_optimizer import BaseOptimizer
from udao_spark.optimizer.moo_algos.evo_optimizer import EvoOptimizer
from udao_spark.optimizer.moo_algos.hierarchical_moo_with_constraints import (
    Hierarchical_MOO_with_Constraints,
)
from udao_spark.optimizer.moo_algos.ws_optimizer import WSOptimizer
from udao_spark.optimizer.utils import (
    Model,
    even_weights,
    save_json,
    save_results,
    weighted_utopia_nearest,
)
from udao_spark.utils.logging import logger
from udao_spark.utils.monitor import DivAndConqMonitor, UdaoMonitor
from udao_trace.parser.spark_parser import THETA_C, THETA_P
from udao_trace.utils import JsonHandler
from udao_trace.utils.interface import VarTypes


class HierarchicalOptimizer(BaseOptimizer):
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        """
        extract the non_decision dict to a DataFrame
        """
        df = pd.DataFrame.from_dict(non_decision_input, orient="index")
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].str.split("-").str[-1].astype(int)
        df.set_index("id", inplace=True, drop=False)
        df.sort_index(inplace=True)
        return df

    def get_objective_values_mlp_arr(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
        place: str = "",
    ) -> np.ndarray:
        tabular_features = th.cat([non_decision_tabular_features, theta], dim=1)
        objs = self.predict_objectives_mlp(graph_embeddings, tabular_features).numpy()
        obj_io = objs[:, 1]
        obj_ana_lat = objs[:, 2]
        theta_c_min, theta_c_max = self.theta_minmax["c"]
        k1_min, k2_min, k3_min = theta_c_min[:3]
        k1_max, k2_max, k3_max = theta_c_max[:3]
        k1 = (theta[:, 0].numpy() - k1_min) * (k1_max - k1_min) + k1_min
        k2 = (theta[:, 1].numpy() - k2_min) * (k2_max - k2_min) + k2_min
        k3 = (theta[:, 2].numpy() - k3_min) * (k3_max - k3_min) + k3_min
        objs_dict = self.summarize_obj(k1, k2, k3, obj_ana_lat, obj_io)

        obj_cost_w_io = objs_dict["ana_cost_w_io"]

        return np.vstack((obj_ana_lat, obj_cost_w_io)).T

    def get_objective_values_ag_arr(
        self,
        graph_embeddings: np.ndarray,
        non_decision_df: pd.DataFrame,
        sampled_theta: np.ndarray,
        model_name: Dict[str, str],
        cost_choice: str = "ana_cost_w_io",
    ) -> np.ndarray:
        start_time_ns = time.perf_counter_ns()
        objs = self.ag_ms.predict_with_ag(
            self.bm,
            self.current_target_template,
            graph_embeddings,
            non_decision_df,
            self.decision_variables,
            sampled_theta,
            model_name,
        )
        end_time_ns = time.perf_counter_ns()
        logger.info(
            f"takes {(end_time_ns - start_time_ns) / 1e6} ms "
            f"to run {len(sampled_theta)} theta"
        )
        objs_dict = self.summarize_obj(
            sampled_theta[:, 0],
            sampled_theta[:, 1],
            sampled_theta[:, 2],
            np.array(objs["ana_latency_s"]),
            np.array(objs["io_mb"]),
        )

        ana_latency = objs["ana_latency_s"]
        ana_cost_w_io = objs_dict[cost_choice]

        return np.vstack((ana_latency, ana_cost_w_io)).T

    def solve(
        self,
        template: str,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
        ag_model: Dict = dict(),
        algo: str = "ws",
        save_data: bool = False,
        query_id: Optional[str] = None,
        sample_mode: str = "",
        param_meta: Dict[str, int] = dict(),
        time_limit: int = -1,
        is_oracle: bool = False,
        save_data_header: str = "./output",
        is_query_control: bool = False,
        benchmark: str = "tpch",
        selected_features: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        if not self.verbose:
            logger.setLevel("ERROR")

        self.current_target_template = template

        # initial a monitor
        start_compute_non_decision = time.time()
        monitor = DivAndConqMonitor() if "hmooc" in algo else UdaoMonitor()
        non_decision_dict = self.extract_data_and_compute_non_decision_features(
            monitor, non_decision_input
        )
        non_decision_df = non_decision_dict["non_decision_df"]
        graph_embeddings = non_decision_dict["graph_embeddings"]
        non_decision_tabular_features = non_decision_dict[
            "non_decision_tabular_features"
        ]
        # checking the monitor is working:
        if (
            monitor.input_extraction_ms <= 0
            or monitor.graph_embedding_computation_ms <= 0
        ):
            raise Exception("Monitor is not working properly!")
        logger.info(
            "input_extraction: {}ms, graph_embedding_computation: {}ms".format(
                monitor.input_extraction_ms, monitor.graph_embedding_computation_ms
            )
        )

        tc_compute_non_decision = time.time() - start_compute_non_decision
        logger.info("graph_embeddings shape: %s", graph_embeddings.shape)
        logger.warning(
            "non_decision_tabular_features is only used for MLP inferencing, shape: %s",
            non_decision_tabular_features.shape,
        )

        start_moo_compile_time_opt = time.time()
        n_subQs = len(non_decision_input)
        len_theta_c = len(self.theta_ktype["c"])
        len_theta_p = len(self.theta_ktype["p"])
        len_theta_s = len(self.theta_ktype["s"])
        len_theta_per_subQ = len_theta_c + len_theta_p + len_theta_s

        if "hmooc" in algo:
            join_ids = []
            for qs_id, v in non_decision_input.items():
                if ".Join" in JsonHandler.dump_to_string(v["qs_lqp"]):
                    join_ids.append(int(qs_id.split("-")[1]))
            if self.verbose:
                print(f"template: {template}: join_ids is {join_ids}")
            objs, conf = self._hmooc(
                len_theta_per_subQ,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_subQs,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                sample_mode,
                param_meta["n_c_samples"],
                param_meta["n_p_samples"],
                is_oracle,
                save_data_header,
                benchmark=benchmark,
                join_ids=join_ids,
                selected_features=selected_features,
            )

        elif algo == "evo":
            objs, conf = self._evo(
                len_theta_per_subQ,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_subQs,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                param_meta["pop_size"],
                param_meta["nfe"],
                time_limit,
                save_data_header=save_data_header,
                is_query_control=is_query_control,
                is_oracle=is_oracle,
            )

        elif algo == "ws":
            objs, conf = self._ws(
                len_theta_per_subQ,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_subQs,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                param_meta["n_samples"],  # n_samples_per_param
                param_meta["n_ws"],
                time_limit,
                save_data_header=save_data_header,
                is_query_control=is_query_control,
                is_oracle=is_oracle,
            )

        elif algo == "ppf":
            objs, conf = self._ppf(
                len_theta_per_subQ,
                graph_embeddings,
                non_decision_df,
                non_decision_tabular_features,
                n_subQs,
                seed,
                use_ag,
                ag_model,
                algo,
                query_id,
                save_data,
                param_meta["n_process"],
                param_meta["n_grids"],
                param_meta["n_max_iters"],
                time_limit,
                is_query_control=is_query_control,
                save_data_header=save_data_header,
                is_oracle=is_oracle,
            )
        else:
            raise Exception(f"Algorithm {algo} is not supported!")

        tc_moo_compile_time_opt = time.time() - start_moo_compile_time_opt
        tc_end_to_end = time.time() - start_compute_non_decision
        time_cost_dict = {
            "compute_non_decision": tc_compute_non_decision,
            f"{algo}_compile_time_opt": tc_moo_compile_time_opt,
            "end_to_end": tc_end_to_end,
        }

        algo_setting = "_".join(str(v) for k, v in param_meta.items())

        if use_ag:
            model_name = "ag"
        else:
            model_name = "mlp"
        if "hmooc" in algo:
            # algo = div_and_conq_moo%GD
            dag_opt_algo = algo.split("%")[1]
            data_path = (
                f"{save_data_header}/{benchmark}/query_control_{is_query_control}/latest_model_{self.device.type}/{model_name}/oracle_{is_oracle}/{algo}/{algo_setting}/time_{time_limit}/"
                f"query_{query_id}_n_{n_subQs}/{sample_mode}/{dag_opt_algo}"
            )
        else:
            data_path = (
                f"{save_data_header}/{benchmark}/query_control_{is_query_control}/latest_model_{self.device.type}/{model_name}/oracle_{is_oracle}/{algo}/{algo_setting}/time_{time_limit}/"
                f"query_{query_id}_n_{n_subQs}"
            )
        save_json(data_path, time_cost_dict, mode="end_to_end")

        monitor.save_to_file(
            file_path=f"{save_data_header}/monitor/{self.bm}_{query_id}_{algo}_add_customized_params.json"
        )

        logger.info(f"conf: {conf}")
        logger.info(f"objs: {objs}")
        return objs, conf, tc_end_to_end

    def _hmooc(
        self,
        len_theta_per_subQ: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_subQs: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        sample_mode: str,
        n_c_samples: int,
        n_p_samples: int,
        is_oracle: bool,
        save_data_header: str,
        benchmark: str,
        join_ids: List[int],
        selected_features: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # return the pareto set
        start = time.time()
        theta_c: Union[th.Tensor, np.ndarray]
        theta_p: Union[th.Tensor, np.ndarray]
        theta_s: Union[th.Tensor, np.ndarray]
        # algo = div_and_conq_moo%GD
        dag_opt_algo = algo.split("%")[1]

        theta_c, theta_p, theta_s = self.get_initial_theta_c_p_s_samples(
            use_ag=use_ag,
            n_c_samples=n_c_samples,
            n_p_samples=n_p_samples,
            sample_mode=sample_mode,
            save_data_header=save_data_header,
            seed=seed,
            selected_features=selected_features,
        )

        # len_theta_per_qs = theta_c.shape[1] + theta_p.shape[1] + theta_s.shape[1]
        n_clusters = 10 if theta_c.shape[0] > 10 else theta_c.shape[0]

        non_decision_features: Union[th.Tensor, pd.DataFrame]
        obj_model: Union[
            Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
            Callable[
                [np.ndarray, pd.DataFrame, np.ndarray, Dict[str, str]], np.ndarray
            ],
        ]
        if use_ag:
            obj_model = self.get_objective_values_ag_arr
            non_decision_features = non_decision_df
        else:
            obj_model = self.get_objective_values_mlp_arr
            non_decision_features = non_decision_tabular_features

        hmooc = Hierarchical_MOO_with_Constraints(
            n_subQs=n_subQs,
            theta_c_samples=theta_c,
            theta_p_samples=theta_p,
            theta_s_samples=theta_s,
            seed=0,
        )
        hmooc.model_setup(
            graph_embeddings=graph_embeddings,
            non_decision_tabular_features=non_decision_features,
            obj_model=obj_model,
            use_ag=use_ag,
            ag_model=ag_model,
        )
        hmooc.hmooc_setup(
            theta_sample_mode=sample_mode,
            theta_sample_funcs=self.sample_theta_x,
            n_clusters=n_clusters,
            cross_location=3,
            dag_opt_algo=dag_opt_algo,
        )

        po_objs, po_conf, model_infer_info, total_time_profile = hmooc.solve()
        time_cost_moo_algo = time.time() - start
        if self.verbose:
            print(f"query id is {query_id}")
            print(f"FUNCTION: time cost of div_and_conq_moo is: {time_cost_moo_algo}")
            print(
                f"The number of Pareto solutions in the DAG opt method"
                f" {dag_opt_algo} is: "
                f"{np.unique(po_objs, axis=0).shape[0]}"
            )
            print(
                f"The Pareto solutions in the DAG opt method"
                f" {dag_opt_algo} is: "
                f"{np.unique(po_objs, axis=0)}"
            )

        start_rec = time.time()

        # added by chenghao to get the PO solutions with preferences directly
        fine_conf_qs_raw = (
            self.sc.construct_configuration(
                po_conf.reshape(-1, len_theta_per_subQ).copy()
            )
            if use_ag
            else self.sc.construct_configuration_from_norm(
                po_conf.reshape(-1, len_theta_per_subQ).copy()
            )
        )
        po_conf_fine_list = []
        po_conf_agg_list = []
        s3 = "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold"
        s4 = "spark.sql.autoBroadcastJoinThreshold"
        s5 = "spark.sql.shuffle.partitions"
        knob_names = self.sc.knob_names
        theta_c_names = knob_names[: len(THETA_C)]
        theta_p_names = knob_names[len(THETA_C) : len(THETA_C) + len(THETA_P)]
        theta_s_names = knob_names[len(THETA_C) + len(THETA_P) :]
        for i in range(po_conf.shape[0]):
            fine_conf = {}
            fine_conf["theta_c"] = {
                k: v
                for k, v in zip(
                    theta_c_names, fine_conf_qs_raw[i * n_subQs][: len(THETA_C)]
                )
            }
            fine_conf["runtime_theta"] = {}
            for j in range(n_subQs):
                qs_theta_p = {
                    k: v
                    for k, v in zip(
                        theta_p_names,
                        fine_conf_qs_raw[i * n_subQs + j][
                            len(THETA_C) : len(THETA_C) + len(THETA_P)
                        ],
                    )
                }
                qs_theta_s = {
                    k: v
                    for k, v in zip(
                        theta_s_names,
                        fine_conf_qs_raw[i * n_subQs + j][
                            len(THETA_C) + len(THETA_P) :
                        ],
                    )
                }

                fine_conf["runtime_theta"][f"qs{j}"] = {
                    "theta_p": qs_theta_p,
                    "theta_s": qs_theta_s,
                }
            theta = {
                k: v for k, v in zip(self.sc.knob_names, fine_conf_qs_raw[i * n_subQs])
            }
            if len(join_ids) > 0:
                theta[s3] = "{}MB".format(
                    min(
                        int(fine_conf["runtime_theta"][f"qs{ji}"]["theta_p"][s3][:-2])
                        for ji in join_ids
                    )
                )
                theta[s4] = "{}MB".format(
                    max(
                        25,  # lower bounded by 25MB to avoid missing BHJ for input QSs
                        min(
                            int(
                                fine_conf["runtime_theta"][f"qs{ji}"]["theta_p"][s4][
                                    :-2
                                ]
                            )
                            for ji in join_ids
                        ),
                    )
                )
                theta[s5] = str(
                    max(
                        int(fine_conf["runtime_theta"][f"qs{ji}"]["theta_p"][s5])
                        for ji in join_ids
                    )
                )

            po_conf_fine_list.append(fine_conf)
            po_conf_agg_list.append(theta)

        pref2theta = {}
        pref2theta_agg = {}
        for pref in [
            (0, 1),
            (0.1, 0.9),
            (0.5, 0.5),
            (0.9, 0.1),
            (1, 0),
        ]:
            objs, conf_fine = weighted_utopia_nearest(
                po_objs, np.array(po_conf_fine_list), np.array(pref)
            )
            objs, conf_agg = weighted_utopia_nearest(
                po_objs, np.array(po_conf_agg_list), np.array(pref)
            )
            pref2theta[f"{pref[0]:.1f}_{pref[1]:.1f}"] = conf_fine
            pref2theta_agg[f"{pref[0]:.1f}_{pref[1]:.1f}"] = conf_agg
            if self.verbose:
                print(f"weights: {pref}, objs: {objs}, conf: {conf_agg}")

        conf_qs0 = po_conf[:, :len_theta_per_subQ].reshape(-1, len_theta_per_subQ)
        conf_all_qs = np.vstack(np.split(po_conf, n_subQs, axis=1))
        if use_ag:
            conf_raw = self.sc.construct_configuration(conf_qs0.copy()).reshape(
                -1, len_theta_per_subQ
            )
            conf_raw_all = self.sc.construct_configuration(conf_all_qs.copy()).reshape(
                -1, len_theta_per_subQ
            )
            if n_c_samples == 1:
                data_path = (
                    f"{save_data_header}/{benchmark}/query_control_False/latest_model_{self.device.type}/"
                    f"ag/oracle_{is_oracle}/{algo}/{n_c_samples}_{n_p_samples}/time_-1/"
                    f"query_{query_id}_n_{n_subQs}/{sample_mode}/{dag_opt_algo}/"
                    f"c_{conf_raw_all[0, 0]}_{conf_raw_all[0, 1]}_{conf_raw_all[0, 2]}"
                )
            else:
                data_path = (
                    f"{save_data_header}/{benchmark}/query_control_False/latest_model_{self.device.type}/ag/oracle_{is_oracle}/{algo}/{n_c_samples}_{n_p_samples}/time_-1/"
                    f"query_{query_id}_n_{n_subQs}/{sample_mode}/{dag_opt_algo}"
                )
        else:
            conf_raw = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                -1, len_theta_per_subQ
            )
            conf_raw_all = self.sc.construct_configuration_from_norm(
                conf_all_qs
            ).reshape(-1, len_theta_per_subQ)
            if n_c_samples == 1:
                data_path = (
                    f"{save_data_header}/{benchmark}/query_control_False/latest_model_{self.device.type}/"
                    f"mlp/oracle_{is_oracle}/{algo}/{n_c_samples}_{n_p_samples}/time_-1/"
                    f"query_{query_id}_n_{n_subQs}/{sample_mode}/{dag_opt_algo}/"
                    f"c_{conf_raw_all[0, 0]}_{conf_raw_all[0, 1]}_{conf_raw_all[0, 2]}"
                )
            else:
                data_path = (
                    f"{save_data_header}/{benchmark}/query_control_False/latest_model_{self.device.type}/mlp/oracle_{is_oracle}/{algo}/{n_c_samples}_{n_p_samples}/time_-1/"
                    f"query_{query_id}_n_{n_subQs}/{sample_mode}/{dag_opt_algo}"
                )

        # placeholder to compute WUN time.
        weighted_utopia_nearest(po_objs, np.array(po_conf_agg_list))
        tc_po_rec = time.time() - start_rec
        if self.verbose:
            print(f"FUNCTION: time cost of {algo} with WUN " f"is: {tc_po_rec}")

        time_cost_moo_total = time.time() - start
        time_cost_dict = {
            f"{algo}": time_cost_moo_algo,
            "wun": tc_po_rec,
            "moo_with_rec": time_cost_moo_total,
        }
        if save_data:
            save_results(data_path, po_objs, mode="F")
            save_results(data_path, conf_raw, mode="Theta")
            save_results(data_path, conf_raw_all, mode="Theta_all")

            save_results(data_path, model_infer_info, mode="model_infer_info")
            save_json(data_path, time_cost_dict, mode="time_cost_json")
            save_json(data_path, total_time_profile, mode="time_profile")

            JsonHandler.dump_to_file(pref2theta, f"{data_path}/pref2theta.json", 2)
            JsonHandler.dump_to_file(
                pref2theta_agg, f"{data_path}/pref2theta_agg.json", 2
            )

        return (
            po_objs,
            np.array(po_conf_agg_list),
        )

    def _evo(
        self,
        len_theta_per_qs: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_stages: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        pop_size: int,
        nfe: int,
        time_limit: int,
        save_data_header: str,
        is_query_control: bool,
        is_oracle: bool,
        weights: np.ndarray = np.array([1.0, 1.0]),
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = time.time()
        if use_ag:
            normalize = False
        else:
            normalize = True
        theta_s_samples = self.sample_theta_x(
            1,
            "s",
            seed + 2 if seed is not None else None,
            selected_features=None,
            normalize=normalize,
        )
        theta_s: Union[th.Tensor, np.ndarray]
        if use_ag:
            theta_s = theta_s_samples
            theta_minmax = copy.deepcopy(self.theta_minmax)
            theta_minmax["s"][0][:] = theta_s[0]
            theta_minmax["s"][1][:] = theta_s[0]
            theta_type = self.theta_ktype
        else:
            theta_s = th.tensor(theta_s_samples, dtype=th.float32)

            theta_minmax = {
                k: (
                    np.zeros_like(v[0]).astype(float),
                    np.ones_like(v[1]).astype(float),
                )
                for k, v in self.theta_minmax.items()
            }
            theta_minmax["s"][0][:] = theta_s[0]
            theta_minmax["s"][1][:] = theta_s[0]
            theta_type = {
                k: [VarTypes.FLOAT] * len(v) for k, v in self.theta_ktype.items()
            }
        non_decision_features: Union[th.Tensor, pd.DataFrame]
        obj_model: Union[
            Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
            Callable[
                [np.ndarray, pd.DataFrame, np.ndarray, Dict[str, str]], np.ndarray
            ],
        ]
        if use_ag:
            obj_model = self.get_objective_values_ag_arr
            non_decision_features = non_decision_df
        else:
            obj_model = self.get_objective_values_mlp_arr
            non_decision_features = non_decision_tabular_features

        time_limit = time_limit
        evo = EvoOptimizer(
            query_id=query_id,
            n_stages=n_stages,
            graph_embeddings=graph_embeddings,
            non_decision_tabular_features=non_decision_features,
            obj_model=obj_model,
            params=EvoOptimizer.Params(
                pop_size=pop_size,
                nfe=nfe,
                fix_randomness_flag=True,
                time_limit=time_limit,
            ),
            use_ag=use_ag,
            ag_model=ag_model,
            theta_minmax=theta_minmax,
            theta_ktype=theta_type,
            is_query_control=is_query_control,
        )
        po_objs, po_conf, model_infer_info = evo.solve()
        time_cost_moo_algo = time.time() - start
        print(f"FUNCTION: time cost of div_and_conq_moo is: {time_cost_moo_algo}")
        print(
            f"The number of Pareto solutions in the {algo} "
            f"is {np.unique(po_objs, axis=0).shape[0]}"
        )
        print()

        start_rec = time.time()
        if -1 in po_objs:  # time out
            po_objs = np.array(po_objs)
            conf2 = np.array(po_conf)
            conf_raw_all = np.array(po_conf)
        else:
            conf_qs0 = po_conf[:, :len_theta_per_qs].reshape(-1, len_theta_per_qs)
            if po_conf.shape[1] == len_theta_per_qs:
                conf_all_qs = po_conf
            else:
                conf_all_p_s = np.vstack(np.split(po_conf[:, 8:], n_stages, axis=1))
                conf_all_c = np.tile(po_conf[:, :8], (n_stages, 1))
                conf_all_qs = np.hstack((conf_all_c, conf_all_p_s))
            if use_ag:
                conf2 = self.sc.construct_configuration(conf_qs0.astype(float)).reshape(
                    -1, len_theta_per_qs
                )
                conf_raw_all = self.sc.construct_configuration(
                    conf_all_qs.astype(float)
                ).reshape(-1, len_theta_per_qs)

            else:
                conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                    -1, len_theta_per_qs
                )
                conf_raw_all = self.sc.construct_configuration_from_norm(
                    conf_all_qs
                ).reshape(-1, len_theta_per_qs)

        if use_ag:
            data_path = (
                f"{save_data_header}/query_control_{is_query_control}/latest_model_{self.device.type}/ag/oracle_{is_oracle}/{algo}/{pop_size}_{nfe}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )
        else:
            data_path = (
                f"{save_data_header}/query_control_{is_query_control}/latest_model_{self.device.type}/mlp/oracle_{is_oracle}/{algo}/{pop_size}_{nfe}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )

        # add WUN
        objs, conf = weighted_utopia_nearest(po_objs, conf2, weights)
        tc_po_rec = time.time() - start_rec
        print(f"FUNCTION: time cost of {algo} with WUN " f"is: {tc_po_rec}")

        time_cost_moo_total = time.time() - start
        time_cost_dict = {
            f"{algo}": time_cost_moo_algo,
            "wun": tc_po_rec,
            "moo_with_rec": time_cost_moo_total,
        }

        if save_data:
            save_results(data_path, po_objs, mode="F")
            save_results(data_path, conf2, mode="Theta")
            save_results(data_path, conf_raw_all, mode="Theta_all")
            # save_results(data_path, np.array([time_cost]), mode="time")
            save_results(data_path, np.array(model_infer_info), mode="model_infer_info")
            save_json(data_path, time_cost_dict, mode="time_cost_json")

        return objs, conf

    def _ws(
        self,
        len_theta_per_qs: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_stages: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        n_samples_per_param: int,
        n_ws: int,
        time_limit: int,
        save_data_header: str,
        is_query_control: bool,
        is_oracle: bool,
        weights: np.ndarray = np.array([1.0, 1.0]),
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = time.time()
        if use_ag:
            normalize = False
        else:
            normalize = True
        theta_s_samples = self.sample_theta_x(
            1,
            "s",
            seed + 2 if seed is not None else None,
            selected_features=None,
            normalize=normalize,
        )
        theta_s: Union[np.ndarray, th.Tensor]
        if use_ag:
            theta_s = theta_s_samples
            c_vars = [
                IntegerVariable(1, 5),
                IntegerVariable(1, 4),
                IntegerVariable(4, 16),
                IntegerVariable(1, 4),
                IntegerVariable(0, 5),
                BoolVariable(),
                BoolVariable(),
                IntegerVariable(50, 75),
            ]
            p_vars = [
                IntegerVariable(0, 5),
                IntegerVariable(1, 6),
                IntegerVariable(0, 32),
                IntegerVariable(0, 32),
                IntegerVariable(2, 50),
                IntegerVariable(0, 4),
                IntegerVariable(20, 80),
                IntegerVariable(0, 4),
                IntegerVariable(0, 4),
            ]
            s_vars = [IntegerVariable(x, x) for x in theta_s[0].tolist()]
        else:
            theta_s = th.tensor(theta_s_samples, dtype=th.float32)
            c_vars = [FloatVariable(0, 1)] * len(self.theta_ktype["c"])
            p_vars = [FloatVariable(0, 1)] * len(self.theta_ktype["p"])
            s_vars = [FloatVariable(x, x) for x in theta_s[0].numpy().tolist()]

        non_decision_features: Union[th.Tensor, pd.DataFrame]
        obj_model: Union[
            Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
            Callable[
                [np.ndarray, pd.DataFrame, np.ndarray, Dict[str, str]], np.ndarray
            ],
        ]
        if use_ag:
            obj_model = self.get_objective_values_ag_arr
            non_decision_features = non_decision_df
        else:
            obj_model = self.get_objective_values_mlp_arr
            non_decision_features = non_decision_tabular_features

        time_limit = time_limit
        # weights
        n_objs = 2
        ws_steps = 1 / (int(n_ws) - 1)
        ws_pairs = even_weights(ws_steps, n_objs)
        ws = WSOptimizer(
            query_id=query_id,
            n_stages=n_stages,
            graph_embeddings=graph_embeddings,
            non_decision_tabular_features=non_decision_features,
            obj_model=obj_model,
            params=WSOptimizer.Params(
                n_samples_per_param=n_samples_per_param,
                ws_pairs=ws_pairs,
                time_limit=time_limit,
            ),
            c_vars=c_vars,
            p_vars=p_vars,
            s_vars=s_vars,
            use_ag=use_ag,
            ag_model=ag_model,
            is_query_control=is_query_control,
        )
        po_objs, po_conf, model_infer_info = ws.solve()
        time_cost_moo_algo = time.time() - start
        print(f"FUNCTION: time cost of {algo} is: {time_cost_moo_algo}")
        print(
            f"The number of Pareto solutions in the {algo} "
            f"is {np.unique(po_objs, axis=0).shape[0]}"
        )
        print()

        start_rec = time.time()
        if -1 in po_objs:  # time out
            po_objs = np.array(po_objs)
            conf2 = np.array(po_conf)
            conf_raw_all = np.array(po_conf)
        else:
            conf_qs0 = po_conf[:, :len_theta_per_qs].reshape(-1, len_theta_per_qs)
            if po_conf.shape[1] == len_theta_per_qs:
                conf_all_qs = po_conf
            else:
                conf_all_p_s = np.vstack(np.split(po_conf[:, 8:], n_stages, axis=1))
                conf_all_c = np.tile(po_conf[:, :8], (n_stages, 1))
                conf_all_qs = np.hstack((conf_all_c, conf_all_p_s))
            if use_ag:
                conf2 = self.sc.construct_configuration(conf_qs0.astype(float)).reshape(
                    -1, len_theta_per_qs
                )
                conf_raw_all = self.sc.construct_configuration(
                    conf_all_qs.astype(float)
                ).reshape(-1, len_theta_per_qs)

            else:
                conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                    -1, len_theta_per_qs
                )
                conf_raw_all = self.sc.construct_configuration_from_norm(
                    conf_all_qs
                ).reshape(-1, len_theta_per_qs)

        if use_ag:
            data_path = (
                f"{save_data_header}/query_control_{is_query_control}/latest_model_{self.device.type}/ag/oracle_{is_oracle}/{algo}/{n_samples_per_param}_{n_ws}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )
        else:
            data_path = (
                f"{save_data_header}/query_control_{is_query_control}/latest_model_{self.device.type}/mlp/oracle_{is_oracle}/{algo}/{n_samples_per_param}_{n_ws}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )

        # add WUN
        objs, conf = weighted_utopia_nearest(po_objs, conf2, weights)
        tc_po_rec = time.time() - start_rec
        print(f"FUNCTION: time cost of {algo} with WUN " f"is: {tc_po_rec}")

        time_cost_moo_total = time.time() - start
        time_cost_dict = {
            f"{algo}": time_cost_moo_algo,
            "wun": tc_po_rec,
            "moo_with_rec": time_cost_moo_total,
        }

        if save_data:
            save_results(data_path, po_objs, mode="F")
            save_results(data_path, conf2, mode="Theta")
            save_results(data_path, conf_raw_all, mode="Theta_all")
            save_results(data_path, np.array(model_infer_info), mode="model_infer_info")
            save_json(data_path, time_cost_dict, mode="time_cost_json")

        return objs, conf

    def _ppf(
        self,
        len_theta_per_qs: int,
        graph_embeddings: th.Tensor,
        non_decision_df: pd.DataFrame,
        non_decision_tabular_features: th.Tensor,
        n_stages: int,
        seed: Optional[int],
        use_ag: bool,
        ag_model: Dict[str, str],
        algo: str,
        query_id: Optional[str],
        save_data: bool,
        n_process: int,
        n_grids: int,
        n_max_iters: int,
        time_limit: int,
        is_query_control: bool,
        save_data_header: str,
        is_oracle: bool,
        weights: np.ndarray = np.array([1.0, 1.0]),
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        start = time.time()
        if use_ag:
            normalize = False
        else:
            normalize = True
        theta_s_samples = self.sample_theta_x(
            1,
            "s",
            seed + 2 if seed is not None else None,
            selected_features=None,
            normalize=normalize,
        )
        theta_s: Union[np.ndarray, th.Tensor]
        if use_ag:
            theta_s = theta_s_samples
            c_vars = [
                IntegerVariable(1, 5),
                IntegerVariable(1, 4),
                IntegerVariable(4, 16),
                IntegerVariable(1, 4),
                IntegerVariable(0, 5),
                BoolVariable(),
                BoolVariable(),
                IntegerVariable(50, 75),
            ]
            p_vars = [
                IntegerVariable(0, 5),
                IntegerVariable(1, 6),
                IntegerVariable(0, 32),
                IntegerVariable(0, 32),
                IntegerVariable(2, 50),
                IntegerVariable(0, 4),
                IntegerVariable(20, 80),
                IntegerVariable(0, 4),
                IntegerVariable(0, 4),
            ]
            s_vars = [IntegerVariable(x, x) for x in theta_s[0].tolist()]
        else:
            theta_s = th.tensor(theta_s_samples, dtype=th.float32)
            c_vars = [FloatVariable(0, 1)] * len(self.theta_ktype["c"])
            p_vars = [FloatVariable(0, 1)] * len(self.theta_ktype["p"])
            s_vars = [FloatVariable(x, x) for x in theta_s[0].numpy().tolist()]

        p_s_vars = p_vars + s_vars
        c_vars_dict: Dict[str, Variable] = {
            f"k{i + 1}": var for i, var in enumerate(c_vars)
        }
        p_s_vars_dict: Dict[str, Variable]
        if is_query_control:
            p_s_vars_dict = {f"s{i + 1}": var for i, var in enumerate(p_s_vars)}
            variables = {**c_vars_dict, **p_s_vars_dict}
            assert len(variables) == (len(c_vars) + len(p_s_vars))
        else:
            p_s_vars_dict = {
                f"qs{qs_id}_s{i + 1}": var
                for qs_id in range(n_stages)
                for i, var in enumerate(p_s_vars)
            }
            variables = {**c_vars_dict, **p_s_vars_dict}
            assert len(variables) == (len(c_vars) + n_stages * len(p_s_vars))

        so = RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=10000))

        non_decision_features: Union[th.Tensor, pd.DataFrame]
        obj_model: Union[
            Callable[
                [np.ndarray, pd.DataFrame, np.ndarray, Dict[str, str]], np.ndarray
            ],
            Callable[[th.Tensor, th.Tensor, th.Tensor, str], np.ndarray],
        ]
        if use_ag:
            obj_model = self.get_objective_values_ag_arr
            non_decision_features = non_decision_df
        else:
            obj_model = self.get_objective_values_mlp_arr
            non_decision_features = non_decision_tabular_features

        model = Model(
            n_stages=n_stages,
            len_theta_c=len(c_vars),
            len_theta_p=len(p_vars),
            len_theta_s=len(s_vars),
            graph_embeddings=graph_embeddings,
            non_decision_tabular_features=non_decision_features,
            obj_model=obj_model,
            use_ag=use_ag,
            ag_model=ag_model,
            is_query_control=is_query_control,
        )
        objectives = [
            Objective("latency", minimize=True, function=model.Obj1),
            Objective("cost", minimize=True, function=model.Obj2),
        ]
        constraints: List[Any] = []
        problem = MOProblem(
            objectives=objectives,
            variables=variables,
            constraints=constraints,
            input_parameters=None,
        )

        # PF-AP
        ppf = ParallelProgressiveFrontier(
            params=ParallelProgressiveFrontier.Params(
                processes=n_process,
                n_grids=n_grids,
                max_iters=n_max_iters,
            ),
            solver=so,
        )
        if time_limit > 0:

            def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
                raise Exception("Timed out!")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(time_limit)

            try:
                po_objs, po_vars = ppf.solve(
                    problem=problem,
                    seed=0,
                )
                signal.alarm(
                    0
                )  ##cancel the timer if the function returned before timeout

            except Exception:
                po_objs, po_vars = np.array([-1]), np.array([-1])
                print(f"Timed out for query {query_id}")
        else:
            po_objs, po_vars = ppf.solve(
                problem=problem,
                seed=0,
            )

        po_conf = np.array([list(x.values()) for x in po_vars])
        time_cost_moo_algo = time.time() - start
        print(f"FUNCTION: time cost of {algo} is: {time_cost_moo_algo}")
        print(
            f"The number of Pareto solutions in the {algo} "
            f"is {np.unique(po_objs, axis=0).shape[0]}"
        )
        print()

        start_rec = time.time()
        if -1 in po_objs:  # time out
            po_objs = np.array(po_objs)
            conf2 = np.array(po_conf)
            conf_raw_all = np.array(po_conf)
        else:
            conf_qs0 = po_conf[:, :len_theta_per_qs].reshape(-1, len_theta_per_qs)
            if po_conf.shape[1] == len_theta_per_qs:
                conf_all_qs = po_conf
            else:
                conf_all_p_s = np.vstack(np.split(po_conf[:, 8:], n_stages, axis=1))
                conf_all_c = np.tile(po_conf[:, :8], (n_stages, 1))
                conf_all_qs = np.hstack((conf_all_c, conf_all_p_s))
            if use_ag:
                conf2 = self.sc.construct_configuration(conf_qs0.astype(float)).reshape(
                    -1, len_theta_per_qs
                )
                conf_raw_all = self.sc.construct_configuration(
                    conf_all_qs.astype(float)
                ).reshape(-1, len_theta_per_qs)

            else:
                conf2 = self.sc.construct_configuration_from_norm(conf_qs0).reshape(
                    -1, len_theta_per_qs
                )
                conf_raw_all = self.sc.construct_configuration_from_norm(
                    conf_all_qs
                ).reshape(-1, len_theta_per_qs)

        if use_ag:
            data_path = (
                f"{save_data_header}/query_control_{is_query_control}/latest_model_{self.device.type}/ag/oracle_{is_oracle}/{algo}/{n_process}_{n_grids}_{n_max_iters}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )
        else:
            data_path = (
                f"{save_data_header}/query_control_{is_query_control}/latest_model_{self.device.type}/mlp/oracle_{is_oracle}/{algo}/{n_process}_{n_grids}_{n_max_iters}/time_{time_limit}/"
                f"query_{query_id}_n_{n_stages}/"
            )

        # add WUN
        objs, conf = weighted_utopia_nearest(po_objs, conf2, weights)
        tc_po_rec = time.time() - start_rec
        print(f"FUNCTION: time cost of {algo} with WUN " f"is: {tc_po_rec}")

        time_cost_moo_total = time.time() - start
        time_cost_dict = {
            f"{algo}": time_cost_moo_algo,
            "wun": tc_po_rec,
            "moo_with_rec": time_cost_moo_total,
        }

        if save_data:
            save_results(data_path, po_objs, mode="F")
            save_results(data_path, conf2, mode="Theta")
            save_results(data_path, conf_raw_all, mode="Theta_all")
            # save_results(data_path, np.array([time_cost]), mode="time")
            save_json(data_path, time_cost_dict, mode="time_cost_json")

        return objs, conf

    def get_initial_theta_c_p_s_samples(
        self,
        use_ag: bool,
        n_c_samples: int,
        n_p_samples: int,
        sample_mode: str,
        save_data_header: str,
        seed: Optional[int],
        selected_features: Optional[Dict[str, List[str]]],
    ) -> Tuple[
        Union[np.ndarray, th.Tensor],
        Union[np.ndarray, th.Tensor],
        Union[np.ndarray, th.Tensor],
    ]:
        if use_ag:
            normalize = False
        else:
            normalize = True

        theta_c: Union[np.ndarray, th.Tensor]
        theta_p: Union[np.ndarray, th.Tensor]
        theta_s: Union[np.ndarray, th.Tensor]
        theta_s_samples = np.array([[20, 2]])
        if use_ag:
            theta_s = theta_s_samples
        else:
            theta_s = th.tensor(theta_s_samples, dtype=th.float32)

        if sample_mode in ["random", "lhs"]:
            theta_c_samples = self.sample_theta_x(
                n_c_samples,
                "c",
                seed if seed is not None else None,
                normalize=normalize,
                mode=sample_mode,
                selected_features=selected_features,
            )
            theta_p_samples = self.sample_theta_x(
                n_p_samples,
                "p",
                seed + 1 if seed is not None else None,
                normalize=normalize,
                mode=sample_mode,
                selected_features=selected_features,
            )
            if use_ag:
                theta_c = theta_c_samples
                theta_p = theta_p_samples
            else:
                theta_c = th.tensor(theta_c_samples, dtype=th.float32)
                theta_p = th.tensor(theta_p_samples, dtype=th.float32)
        elif sample_mode.startswith("grid"):
            if sample_mode == "grid-search":
                if n_c_samples == 512:
                    c_grids = [
                        [1, 2, 4, 5],
                        [1, 4],
                        [4, 16],
                        [1, 4],
                        [0, 5],  # k5 not important
                        [0, 1],
                        [0, 1],
                        [60],  # k8 not important
                    ]

                elif n_c_samples == 256:
                    c_grids = [
                        [1, 5],
                        [1, 4],
                        [4, 16],
                        [1, 4],
                        [0, 5],
                        [0, 1],
                        [0, 1],
                        [50, 75],
                    ]
                elif n_c_samples == 128:
                    c_grids = [
                        [1, 5],
                        [1, 4],
                        [4, 16],
                        [1, 4],
                        [0, 5],
                        [0, 1],
                        [1],
                        [50, 75],
                    ]

                elif n_c_samples == 160:  # 4  2  5  1  2 * 2 = 160
                    # e2e performance and time measurements
                    # 32 - choose top-5 with all high-low combinations
                    # 2^6 64 (all high-low combinations)
                    # 2^7 128 (4 for most important knob k3)
                    # 2^8 256 (4 for top-2 most important knob k3, k1)

                    c_grids = [
                        [1, 2, 3, 5],  # 4
                        [1, 2],  # 2
                        [8, 10, 12, 14, 16],  # 5
                        [2],  # k4
                        [2],
                        [0, 1],  # 2
                        [1],  # need more choices for k7
                        [60],  # 2
                    ]

                elif n_c_samples == 360:  # 4  2  5  1  2 * 2 = 360
                    c_grids = [
                        [1, 2, 3, 4, 5],  # 5
                        [1, 2, 3],  # 3
                        [6, 8, 10, 12, 14, 16],  # 6
                        [2],
                        [2],
                        [0, 1],  # 2
                        [1],
                        [50, 75],  # 2
                    ]

                elif n_c_samples == 64:
                    if "test_end" in save_data_header:
                        # to test for end-to-end
                        c_grids = [
                            [2, 3, 4, 5],
                            [1, 2, 3, 4],
                            [4, 8, 12, 16],
                            [1, 4],
                            [0, 5],
                            [0],
                            [1],
                            [50, 75],
                        ]
                    else:
                        c_grids = [
                            [1, 5],
                            [1, 4],
                            [4, 16],
                            [1, 4],
                            [0, 5],
                            [0],
                            [1],
                            [50, 75],
                        ]
                elif n_c_samples == 36:  # 3  2  3 * 2 (<= 64)
                    c_grids = [
                        [1, 3, 5],  # k1: 3
                        [1, 2],  # k2: 2
                        [8, 12, 16],  # k3: 3
                        [2],  # k4:
                        [2],  # k5
                        [0, 1],  # k6: 2
                        [1],  # k7
                        [60],  # k8
                    ]
                elif n_c_samples == 32:
                    if "test_end" in save_data_header:
                        # to test for end-to-end
                        c_grids = [
                            [2, 3, 4, 5],
                            [1, 2, 3, 4],
                            [4, 8, 12, 16],
                            [1, 4],
                            [5],
                            [0],
                            [1],
                            [50, 75],
                        ]
                    else:
                        c_grids = [
                            [1, 5],
                            [1, 4],
                            [4, 16],
                            [1, 4],
                            [5],
                            [0],
                            [1],
                            [50, 75],
                        ]

                elif n_c_samples == 16:
                    c_grids = [
                        [1, 5],
                        [1, 4],
                        [4, 16],
                        [1, 4],
                        [5],
                        [0],
                        [1],
                        [50],
                    ]
                elif n_c_samples == 1:
                    c_grids = [
                        [1],  # r1: 1, r2: 3, r3: 5, r4: 1
                        [1],  # r1: 1, r2: 1, r3: 4, r4: 4
                        [16],  # r1: 4, r2: 16, r3: 16, r4: 4
                        [1],
                        [5],
                        [0],
                        [1],
                        [75],  ##1,1,16
                    ]
                else:
                    raise Exception(f"n_c_samples {n_c_samples} is not supported!")

                if n_p_samples == 512:
                    p_grids = [
                        [0, 5],
                        [1, 6],
                        [0, 32],
                        [0, 32],
                        [2, 50],
                        [0, 4],
                        [20, 80],
                        [0, 4],
                        [0, 4],
                    ]
                elif n_p_samples == 256:
                    if "test_end_diff_p" in save_data_header:
                        p_grids = [
                            [0],
                            [1, 6],
                            [0, 32],
                            [0, 32],
                            [2, 50],
                            [0, 4],
                            [20, 80],
                            [0, 4],
                            [0, 4],
                        ]
                    else:
                        p_grids = [
                            [0],
                            [1, 6],
                            [0, 32],
                            [0, 32],
                            [2, 50],
                            [0, 4],
                            [20, 80],
                            [0, 4],
                            [0, 4],
                        ]
                elif n_p_samples == 162:  # 3  3  3  3  2 = 162
                    p_grids = [
                        [2],  # default
                        [2],  # default
                        [0, 14, 28],  # 0MB/160MB maxShuffledHashJoinLocalMapThreshold
                        [0, 14, 28],  # 0MB/160MB autoBroadcastJoinThreshold
                        [10, 20, 40],  # 80/160/320 sql.shuffle.partitions
                        [2],
                        [50],  # default
                        [1, 2, 3],
                        [1, 2],  # default
                    ]

                elif n_p_samples == 324:  # 2 3  3  3  3  2 = 162
                    p_grids = [
                        [2, 4],  # default
                        [2],  # default
                        [0, 14, 28],  # 0MB/160MB maxShuffledHashJoinLocalMapThreshold
                        [0, 14, 28],  # 0MB/160MB autoBroadcastJoinThreshold
                        [10, 20, 40],  # 80/160/320 sql.shuffle.partitions
                        [2],
                        [50],  # default
                        [1, 2, 3],
                        [1, 2],  # default
                    ]

                elif n_p_samples == 128:
                    p_grids = [
                        [0],
                        [1, 6],
                        [0, 32],
                        [0, 32],
                        [2, 50],
                        [0, 4],
                        [20, 80],
                        [0],
                        [0, 4],
                    ]
                elif n_p_samples == 64:
                    p_grids = [
                        [0],
                        [1],
                        [0, 32],
                        [0, 32],
                        [2, 50],
                        [0, 4],
                        [20, 80],
                        [0],
                        [0, 4],
                    ]
                elif n_p_samples == 32:
                    if "test_end_diff_p" in save_data_header:
                        p_grids = [
                            [0],
                            [1],
                            [0, 32],
                            [0, 32],
                            [2, 50],
                            [0],
                            [20, 80],
                            [0],
                            [0, 4],
                        ]
                    else:
                        p_grids = [
                            [0],
                            [1],
                            [0, 32],
                            [0, 32],
                            [2, 50],
                            [0],
                            [20, 80],
                            [0],
                            [0, 4],
                        ]
                elif n_p_samples == 24:  # 2  2  2 * 3
                    p_grids = [
                        [2],  # default
                        [2],  # default
                        [0, 14],  # 0MB/140MB maxShuffledHashJoinLocalMapThreshold
                        [0, 14],  # 0MB/140MB autoBroadcastJoinThreshold
                        [10, 20],  # 80/160 sql.shuffle.partitions
                        [2],
                        [50],  # default
                        [1, 2, 3],
                        [2],  # default
                    ]
                elif n_p_samples == 16:
                    p_grids = [
                        [0],
                        [1],
                        [0, 32],
                        [0, 32],
                        [2, 50],
                        [0],
                        [20, 80],
                        [0],
                        [0],
                    ]
                else:
                    raise Exception(f"n_p_samples {n_p_samples} is not supported!")
            elif sample_mode == "grid-adaptive-cut":  # cut at cumulative 5\%
                # k3=[8, 12, 16] & turn on s3 in 81

                # the choices of grid based on the feature importance score (FIS)
                # set default to parameters from the low rank to the high rank
                # that cumulatively sum up to 5% of WMAPE
                #
                # k7, k1, k3, k2 (set k2, k4, k6, k5 and k8 to default)
                # query plan related params: s3, s4
                # other SQL params with FIS: s5, s8, s9, s1 (set s2, s6, s7 to default)
                if n_c_samples not in [54, 90, 150]:
                    raise Exception(
                        f"# of theta_c samples {n_c_samples} "
                        f"is not supported for {sample_mode}!"
                    )
                if n_p_samples not in [27, 81, 243]:
                    raise Exception(
                        f"# of theta_p samples {n_p_samples} "
                        f"is not supported for {sample_mode}!"
                    )
                if n_c_samples == 54:
                    c_grids = [
                        [1, 3, 5],  # k1
                        [1, 2, 3],  # k2
                        [8, 12, 16],  # k3
                        [2],  # k4 - from best practice
                        [2],  # k5 - default: 2
                        [0],  # k6 - set to "0"
                        [0, 1],  # k7
                        [60],  # k8 - default: 60
                    ]
                elif n_c_samples == 90:  # 5 * 3 * 3 * 2 = 90
                    c_grids = [
                        [1, 2, 3, 4, 5],  # k1
                        [1, 2, 3],  # k2
                        [8, 12, 16],  # k3
                        [2],  # k4 - from best practice
                        [2],  # k5 - default: 2
                        [0],  # k6 - set to "0"
                        [0, 1],  # k7
                        [60],  # k8 - default: 60
                    ]
                elif n_c_samples == 150:  # 5 * 5 * 3 * 2 = 150
                    c_grids = [
                        [1, 2, 3, 4, 5],  # k1
                        [1, 2, 3],  # k2
                        [8, 10, 12, 14, 16],  # k3
                        [2],  # k4 - from best practice
                        [2],  # k5 - default: 2
                        [0],  # k6 - set to "0"
                        [0, 1],  # k7
                        [60],  # k8 - default: 60
                    ]
                else:
                    raise Exception(
                        f"# of theta_c samples {n_c_samples} "
                        f"is not supported for {sample_mode}!"
                    )

                # add query plan related params: s4, s3
                # add other params s5, s8, s9, s1 (set s2, s6, s7 to default)
                # for some realistic concerns, we reset the range for
                # s4: [0MB - 280MB] to avoid failures and missing good broadcast
                # s5: [10 - 50] to avoid bad performance within same resource usage
                if n_p_samples == 27:
                    p_grids = [
                        [2],  # s1 <--
                        [2],  # s2 default
                        [0, 14, 28],  # s3: maxShuffledHashJoinLocalMapThreshold
                        [0, 14, 28],  # s4: 10/140/280MB autoBroadcastJoinThreshold
                        [10, 20, 50],  # s5: 80/160/400 sql.shuffle.partitions
                        [2],  # s6 default
                        [50],  # s7: default
                        [2],  # s8: spark.sql.files.maxPartitionBytes
                        [2],  # s9: default
                    ]
                elif n_p_samples == 81:  # 3^4 = 81
                    p_grids = [
                        [2],  # s1 <--
                        [2],  # s2 default
                        [0, 14, 28],  # s3: maxShuffledHashJoinLocalMapThreshold
                        [0, 14, 28],  # s4: 10/140/280MB autoBroadcastJoinThreshold
                        [10, 20, 50],  # s5: 80/160/400 sql.shuffle.partitions
                        [2],  # s6 default
                        [50],  # s7: default
                        [0, 2, 4],  # s8: spark.sql.files.maxPartitionBytes
                        [2],  # s9: default
                    ]
                elif n_p_samples == 243:  # 3^5 = 243
                    p_grids = [
                        [2],  # s1 <--
                        [2],  # s2 default
                        [0, 14, 28],  # s3: maxShuffledHashJoinLocalMapThreshold
                        [0, 14, 28],  # s4: 10/140/280MB autoBroadcastJoinThreshold
                        [10, 20, 50],  # s5: 80/160/400 sql.shuffle.partitions
                        [2],  # s6 default
                        [50],  # s7: default
                        [0, 2, 4],  # s8: spark.sql.files.maxPartitionBytes
                        [0, 2, 4],  # s9
                    ]
            else:
                raise Exception(
                    f"The sample mode {sample_mode} for theta is not supported!"
                )

            if use_ag:
                theta_c = np.array([list(i) for i in itertools.product(*c_grids)])
                theta_p = np.array([list(i) for i in itertools.product(*p_grids)])
            else:
                theta_c_samples = np.array(
                    [list(i) for i in itertools.product(*c_grids)]
                )
                theta_p_samples = np.array(
                    [list(i) for i in itertools.product(*p_grids)]
                )
                c_samples_norm = (theta_c_samples - self.theta_minmax["c"][0]) / (
                    self.theta_minmax["c"][1] - self.theta_minmax["c"][0]
                )
                theta_c = th.tensor(c_samples_norm, dtype=th.float32)
                p_samples_norm = (theta_p_samples - self.theta_minmax["p"][0]) / (
                    self.theta_minmax["p"][1] - self.theta_minmax["p"][0]
                )
                theta_p = th.tensor(p_samples_norm, dtype=th.float32)

        else:
            raise Exception(
                f"The sample mode {sample_mode} for theta is not supported!"
            )

        return theta_c, theta_p, theta_s
