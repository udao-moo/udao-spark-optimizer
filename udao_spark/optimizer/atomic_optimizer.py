import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
from udao.optimization.utils.moo_utils import is_pareto_efficient

from ..utils.constants import THETA_COMPILE
from ..utils.logging import logger
from ..utils.monitor import UdaoMonitor
from .base_optimizer import BaseOptimizer


class AtomicOptimizer(BaseOptimizer):
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        """
        extract the non_decision dict to a DataFrame
        """
        df = pd.DataFrame.from_dict({0: non_decision_input}, orient="index")
        df.index.name = "id"
        df["id"] = df.index
        return df

    def construct_po_confs(self, po_theta: np.ndarray, use_ag: bool) -> np.ndarray:
        non_decision_knobs = [
            v for v in THETA_COMPILE if v not in self.decision_variables
        ]
        n_vars = len(self.decision_variables)
        n_consts = len(non_decision_knobs)
        n_po = len(po_theta)
        po_theta_full = np.hstack([np.zeros((n_po, n_consts)), po_theta])
        if use_ag:
            po_confs = self.sc.construct_configuration(po_theta_full)[:, -n_vars:]
        else:
            po_confs = self.sc.construct_configuration_from_norm(po_theta_full)[
                :, -n_vars:
            ]
        return po_confs

    def solve(
        self,
        template: str,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
        ag_model: Dict[str, str] = dict(),
        sample_mode: str = "random",
        n_samples: int = 1,
        pre_samples: Optional[np.ndarray] = None,
        moo_mode: str = "BF",
        monitor: UdaoMonitor = UdaoMonitor(),
        ercilla: bool = True,
        graph_choice: str = "gtn",
        selected_features: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        non_decision_dict = self.extract_data_and_compute_non_decision_features(
            monitor, non_decision_input, use_ag, ercilla, graph_choice
        )
        non_decision_df = non_decision_dict["non_decision_df"]
        graph_embeddings = non_decision_dict["graph_embeddings"]
        non_decision_tabular_features = non_decision_dict[
            "non_decision_tabular_features"
        ]

        t3 = time.perf_counter_ns()
        if sample_mode in ["random", "lhs"]:
            # for lhs, a general way is to use:
            # sampled_theta = get_lhs_confs(
            #   self.sc, n_samples, seed=seed, normalize=not use_ag).values
            # )
            # since the knobs are all int in our case, we can simplify it to
            # sample via self.sample_theta_all
            sampled_theta = self.sample_theta_all(
                n_samples=n_samples,
                seed=seed,
                selected_features=selected_features,
                normalize=not use_ag,
                mode=sample_mode,
            )[:, -len(self.decision_variables) :]
        elif sample_mode == "preset":
            if pre_samples is None:
                raise ValueError("samples is required for preset sampling")
            sampled_theta = pre_samples
        elif sample_mode == "grid":
            raise NotImplementedError
        else:
            raise ValueError(f"sample_mode {sample_mode} is not supported")
        t4 = time.perf_counter_ns()
        monitor.theta_sampling_ms = (t4 - t3) / 1e6  # monitoring
        monitor.theta_numbers = n_samples  # monitoring
        if self.verbose:
            logger.info(f">> sampled {n_samples} theta in {(t4 - t3) / 1e6} ms")

        if use_ag:
            graph_embeddings = graph_embeddings.detach().cpu()
            objs_dict = self.get_objective_values_ag(
                template,
                graph_embeddings.tile(n_samples, 1).numpy(),
                pd.DataFrame(
                    np.tile(non_decision_df.values, (n_samples, 1)),
                    columns=non_decision_df.columns,
                ),
                sampled_theta,
                ag_model,
            )
        else:
            if non_decision_tabular_features is None:
                raise ValueError(
                    "non_decision_tabular_features is required for MLP inference"
                )
            objs_dict = self.get_objective_values_mlp(
                graph_embeddings.tile(n_samples, 1),
                non_decision_tabular_features.tile(n_samples, 1),
                th.tensor(sampled_theta, dtype=self.dtype),
            )
        lat, cost = self.get_latencies_and_objectives(objs_dict)
        print(lat.sum())
        objs = np.vstack([lat, cost]).T.astype(np.float32)
        t5 = time.perf_counter_ns()
        monitor.model_inference_ms = (t5 - t4) / 1e6  # monitoring
        if self.verbose:
            logger.info(f">> computed objective values in {(t5 - t4) / 1e6} ms")

        if moo_mode == "BF":
            po_mask = is_pareto_efficient(objs)
        else:
            raise ValueError(f"moo_mode {moo_mode} is not supported")
        po_objs = objs[po_mask]
        po_theta = sampled_theta[po_mask]

        t6 = time.perf_counter_ns()
        monitor.pareto_filtering_ms = (t6 - t5) / 1e6  # monitoring
        if self.verbose:
            logger.info(f">> filtered pareto optimal points in {(t6 - t5) / 1e6} ms")

        n_po = len(po_objs)
        if n_po == 0:
            return None, None, -1

        logger.debug(f"po_objs: {po_objs}, po_theta: {po_theta}")
        po_confs = self.construct_po_confs(po_theta, use_ag)
        logger.debug(f"found {len(po_objs)} po points, po_confs: {po_confs}")

        t7 = time.perf_counter_ns()
        monitor.output_preparation_ms += (t7 - t6) / 1e6  # monitoring
        if self.verbose:
            logger.info(f">> constructed configurations in {(t7 - t6) / 1e6} ms")

        return po_objs, po_confs, -1
