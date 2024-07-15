# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 05/02/2024
import signal
import time
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from udao.optimization.concepts import Variable

import udao_spark.optimizer.utils as ut
from udao_spark.optimizer.soo.random_sampler import RandomSampler


class WSOptimizer:
    @dataclass
    class Params:
        n_samples_per_param: int
        "the number of samples per parameter"
        ws_pairs: list
        "a set of weight pairs"
        time_limit: int = -1
        "the time limit of running compile-time optimization (-1: no-limit)"
        verbose: bool = False
        "flag to indicate whether to print more info"

    def __init__(
        self,
        query_id: Optional[str],
        n_stages: int,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: Any,
        obj_model: Callable[[Any, Any, Any, Any], Any],
        c_vars: List[Variable],
        p_vars: List[Variable],
        s_vars: List[Variable],
        params: Params,
        use_ag: bool,
        ag_model: Dict[str, str],
        is_query_control: bool,
    ) -> None:
        self.n_stages = n_stages
        self.query_id = query_id
        self.graph_embeddings = graph_embeddings
        self.non_decision_tabular_features = non_decision_tabular_features
        self.obj_model = obj_model

        self.n_samples = params.n_samples_per_param
        self.ws_pairs = params.ws_pairs
        self.time_limit = params.time_limit

        self.seed = 0
        self.solver = RandomSampler(RandomSampler.Params(self.n_samples, self.seed))

        if is_query_control:
            self.vars = c_vars + p_vars + s_vars
        else:
            self.vars = c_vars + n_stages * (p_vars + s_vars)  # type: ignore
        self.len_theta_c = len(c_vars)
        self.len_theta_p = len(p_vars)
        self.len_theta_s = len(s_vars)

        self.use_ag = use_ag
        self.ag_model = ag_model

        self.is_query_control = is_query_control
        print()

    def solve(self) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        if self.time_limit > 0:

            def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
                raise Exception("Timed out!")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.time_limit)

            try:
                F, Theta, model_infer_info = self.ws_solve()
                signal.alarm(
                    0
                )  ##cancel the timer if the function returned before timeout

            except Exception:
                F, Theta = np.array([-1]), np.array([-1])
                print(f"Timed out for query {self.query_id}")
        else:
            F, Theta, model_infer_info = self.ws_solve()

        return F, Theta, model_infer_info

    def ws_solve(self) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        theta = self.solver._get_input(self.vars)  # type: ignore
        po_obj_list, po_var_list = [], []

        mesh_theta_c: Union[np.ndarray, th.Tensor]
        theta_p_s: Union[np.ndarray, th.Tensor]
        mesh_tabular_features: Union[th.Tensor, pd.DataFrame]
        model_infer_info_list = []

        if self.use_ag:
            assert isinstance(theta, np.ndarray)

            if self.is_query_control:
                assert theta.shape[1] == (
                    self.len_theta_c + self.len_theta_p + self.len_theta_s
                )
                mesh_theta_c = np.tile(theta[:, : self.len_theta_c], (self.n_stages, 1))
                mesh_theta_p_s = np.tile(
                    theta[:, self.len_theta_c :], (self.n_stages, 1)
                )
                theta_all = np.concatenate([mesh_theta_c, mesh_theta_p_s], axis=1)
            else:
                assert theta.shape[1] == (
                    self.len_theta_c
                    + self.n_stages * (self.len_theta_p + self.len_theta_s)
                )
                mesh_theta_c = np.tile(theta[:, : self.len_theta_c], (self.n_stages, 1))
                len_theta_p_s = self.len_theta_p + self.len_theta_s
                theta_p_s = theta[:, self.len_theta_c :].reshape(
                    self.n_stages * self.n_samples, len_theta_p_s
                )
                theta_all = np.concatenate([mesh_theta_c, theta_p_s], axis=1)

            mesh_graph_embeddings = self.graph_embeddings.repeat_interleave(
                self.n_samples, dim=0
            )

            assert isinstance(self.non_decision_tabular_features, pd.DataFrame)
            mesh_tabular_features = self.non_decision_tabular_features.loc[
                np.repeat(self.non_decision_tabular_features.index, self.n_samples)
            ].reset_index(drop=True)
            start_pred = time.time()
            objs = self.obj_model(
                mesh_graph_embeddings.cpu().numpy(),
                mesh_tabular_features,
                theta_all,
                self.ag_model,
            )
            n_evals = theta_all.shape[0]
            tc_pred = time.time() - start_pred
            model_infer_info_list.append([n_evals, tc_pred])

        else:
            # shape: (n_samples/grids * n_objs)
            assert isinstance(self.non_decision_tabular_features, th.Tensor)
            mesh_theta_c = th.tensor(theta[:, : self.len_theta_c]).repeat(
                self.n_stages, 1
            )
            len_theta_p_s = self.len_theta_p + self.len_theta_s
            if self.is_query_control:
                theta_p_s = th.tensor(theta[:, self.len_theta_c :]).repeat(
                    self.n_stages, 1
                )
            else:
                theta_p_s = th.tensor(theta[:, self.len_theta_c :]).reshape(
                    self.n_stages * self.n_samples, len_theta_p_s
                )
            theta_all = th.cat([mesh_theta_c, theta_p_s], dim=1)
            mesh_graph_embeddings = self.graph_embeddings.repeat_interleave(
                self.n_samples, dim=0
            )
            mesh_tabular_features = (
                self.non_decision_tabular_features.repeat_interleave(
                    self.n_samples, dim=0
                )
            )
            objs = self.obj_model(
                mesh_graph_embeddings,
                mesh_tabular_features,
                theta_all.type(th.float32),
                "",
            )
        # shape (n_samples, n_stages)
        latency = np.vstack(np.split(objs[:, 0], self.n_stages)).T
        cost = np.vstack(np.split(objs[:, 1], self.n_stages)).T

        query_latency = np.sum(latency, axis=1)
        query_cost = np.sum(cost, axis=1)
        query_objs = np.vstack((query_latency, query_cost)).T
        # objs = node_model.query_objs(theta, self.n_stages, self.query_id, obj2="obj2")
        assert query_objs.shape[0] == self.n_samples

        # normalization
        objs_min, objs_max = query_objs.min(0), query_objs.max(0)

        if all((objs_min - objs_max) <= 0):
            objs_norm = (query_objs - objs_min) / (objs_max - objs_min)
            for ws in self.ws_pairs:
                # po_ind = self.get_soo_index(objs_norm, ws)
                obj = np.sum(objs_norm * ws, axis=1)
                po_ind = np.argmin(obj)
                po_obj_list.append(query_objs[po_ind])
                po_var_list.append(theta[po_ind])

            # only keep non-dominated solutions
            po, confs = ut.keep_non_dominated(po_obj_list, po_var_list)

            return po, confs, model_infer_info_list
            # return moo_ut._summarize_ret(po_obj_list, po_var_list)
        else:
            raise Exception(
                "Cannot do normalization! Lower bounds of objective values "
                "are higher than their upper bounds."
            )
