# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: EVO optimizer
#
# Created at 05/02/2024

import random
import signal
import time
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from platypus import (
    HUX,
    NSGAII,
    PM,
    SBX,
    Archive,
    BitFlip,
    CompoundOperator,
    GAOperator,
    Integer,
    Problem,
    Real,
    Type,
    nondominated,
)

from udao_trace.utils.interface import VarTypes


class Constant(Type):
    def __init__(self, value: Any) -> None:
        super(Constant, self).__init__()
        self.value = value

    def rand(self) -> Any:
        return self.value

    def __str__(self) -> Any:
        return "Constant(%s)" % (self.value)


global_var_range: np.ndarray
enum_inds: List[Any]
model_infer_info: List[Any]


class EvoOptimizer:
    @dataclass
    class Params:
        pop_size: Optional[int]
        "the population size"
        nfe: Optional[int]
        "the number of function evaluations, i.e. n_iters * pop_size"
        fix_randomness_flag: bool = True
        "flag to indicate whether to fix randomness of initial population"
        inner_algo: str = "NSGA-II"
        "the EVO MOO algorithm"
        time_limit: int = -1
        "the time limit of running compile-time optimization (-1: no-limit)"
        verbose: bool = False
        "flag to indicate whether to print more info"

    def __init__(
        self,
        query_id: Optional[str],
        n_stages: int,
        graph_embeddings: th.Tensor,
        # non_decision_tabular_features: Union[th.Tensor, pd.DataFrame],
        non_decision_tabular_features: Any,
        obj_model: Callable[[Any, Any, Any, Any], Any],
        theta_minmax: dict,
        theta_ktype: dict,
        params: Params,
        use_ag: bool,
        ag_model: Dict[str, str],
        is_query_control: bool,
    ) -> None:
        self.query_id = query_id
        self.n_stages = n_stages
        self.graph_embeddings = graph_embeddings
        self.non_decision_tabular_features = non_decision_tabular_features
        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
        self.seed = 0
        self.theta_minmax = theta_minmax
        self.theta_ktype = theta_ktype
        self.obj_model = obj_model
        self.len_theta_c = len(theta_ktype["c"])
        self.len_theta_p = len(theta_ktype["p"])
        self.len_theta_s = len(theta_ktype["s"])

        self.use_ag = use_ag
        self.ag_model = ag_model

        self.n_objs = 2
        self.n_consts = 0

        self.inner_algo = params.inner_algo
        self.pop_size = params.pop_size
        self.nfe = params.nfe
        self.fix_randomness_flag = params.fix_randomness_flag
        if params.inner_algo == "NSGA-II":
            self.moo = NSGAII
        else:
            raise Exception(f"Algorithm {params.inner_algo} is not supported!")

        self.time_limit = params.time_limit
        self.is_query_control = is_query_control

    def solve(self) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        if self.time_limit > 0:

            def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
                raise Exception("Timed out!")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.time_limit)

            try:
                F, Theta, model_infer_info = self.evo_solve()
                signal.alarm(
                    0
                )  ##cancel the timer if the function returned before timeout

            except Exception:
                F, Theta = np.array([-1]), np.array([-1])
                model_infer_info = [-1]
                print(f"Timed out for query {self.query_id}")
        else:
            F, Theta, model_infer_info = self.evo_solve()

        return F, Theta, model_infer_info

    def evo_solve(self) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """
        solve MOO with NSGA-II algorithm
        :param wl_id: str, workload id, e.g. '1-7'
        :param var_ranges: ndarray(n_vars,), lower and upper var_ranges of
        variables(non-ENUM), and values of ENUM variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return:
                po_objs: ndarray(n_solutions, n_objs), Pareto solutions
                po_vars: ndarray(n_solutions, n_vars), corresponding variable
                values of Pareto solutions
        """

        # global global_var_range
        vars_range = {k: np.vstack(v).T for k, v in self.theta_minmax.items()}
        # shape (n_vars, 2), col1: lower, col2: upper
        # force to use normalized theta
        # vars_range = {k: np.array([]).T for k in self.theta_minmax.keys()}
        global global_var_range
        global model_infer_info

        model_infer_info = []
        # create a new problem
        theta_c_ktypes = self.theta_ktype["c"]
        theta_p_ktypes = self.theta_ktype["p"]
        theta_s_ktypes = self.theta_ktype["s"]

        if self.is_query_control:
            global_var_range = np.array(
                vars_range["c"].tolist()
                + (vars_range["p"].tolist() + vars_range["s"].tolist())
            )
            # here theta_s is constant
            vars_types = theta_c_ktypes + (theta_p_ktypes + theta_s_ktypes)
            n_vars = len(vars_types)
        else:
            global_var_range = np.array(
                vars_range["c"].tolist()
                + (vars_range["p"].tolist() + vars_range["s"].tolist()) * self.n_stages
            )
            # here theta_s is constant
            vars_types = theta_c_ktypes + self.n_stages * (
                theta_p_ktypes + theta_s_ktypes
            )
            n_vars = len(vars_types)

        # class to add all solutions during evolutionary iterations
        class LoggingArchive(Archive):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, *kwargs)
                self.log: List[Any] = []

            def add(self, solution: Any) -> None:
                super().add(solution)
                self.log.append(solution)

        log_archive = LoggingArchive()

        n_objs = self.n_objs
        n_consts = self.n_consts
        self.problem = Problem(n_vars, n_objs, n_consts)

        # set up variable types and constraint types
        flag_mixed_var_type = self.set_var_types(vars_types)
        # pass the functions of objectives and constraints
        self.problem.function = self.problem_def

        # fix randomness
        if self.fix_randomness_flag:
            random.seed(self.seed)

        # required by the Platypus, mixed types must define variator.
        if flag_mixed_var_type:
            if self.use_ag:
                algorithm = self.moo(
                    self.problem,
                    population_size=self.pop_size,
                    variator=GAOperator(HUX(), BitFlip()),
                    archive=log_archive,
                )
            else:
                algorithm = self.moo(
                    self.problem,
                    population_size=self.pop_size,
                    variator=CompoundOperator(SBX(), PM()),
                    archive=log_archive,
                )
        else:
            algorithm = self.moo(
                self.problem, population_size=self.pop_size, archive=log_archive
            )
        algorithm.run(self.nfe)

        # find feasible solutions
        feasible_solutions_algo = [s for s in algorithm.result if s.feasible]
        if len(feasible_solutions_algo) > 0:
            # if the algorithm returns feasible solutions, keep them to further
            # find non-dominated solutions
            feasible_solutions = feasible_solutions_algo
        else:
            # if no feasible solutions are returned by the algorithm
            # (i.e. cannot satisfy constraints), it tries to find feasible solutions
            # among all iterations from log
            feasible_solutions_log = [s for s in log_archive.log if s.feasible]
            feasible_solutions = feasible_solutions_log

        # find non-dominated solutions
        po_objs_list: List[Any] = []
        po_vars_list: List[Any] = []
        if feasible_solutions == []:
            print(f"Evo({self.inner_algo}) cannot find feasible solutions!")
            po_objs_arr, po_confs_arr = np.array([-1]), np.array([-1])
            model_infer_info = [-1]
            return po_objs_arr, po_confs_arr, model_infer_info
        else:
            # find non-dominated solutions
            non_dominated = nondominated(feasible_solutions)
            non_dominated_objs = [
                solution.objectives._data for solution in non_dominated
            ]
            # filter duplicates
            uniq_non_dominated_objs, uniq_non_dominated_index = np.unique(
                np.array(non_dominated_objs), axis=0, return_index=True
            )
            uniq_non_dominated = np.array(non_dominated)[
                uniq_non_dominated_index
            ].tolist()
            print(
                f"the number of non-dominated solutions is "
                f"{len(uniq_non_dominated_objs)}"
            )

            for solution in uniq_non_dominated:
                # Within the internal Platypus library, the INTEGER variable is
                # encoded with binary numbers.
                # Here it uses decode to return back the INTEGER value
                po_vars = [
                    x.decode(y)
                    for [x, y] in zip(self.problem.types, solution.variables)
                ]
                for i in enum_inds:
                    ind = int(po_vars[i])
                    decoded_enum = global_var_range[i][ind]
                    po_vars[i] = decoded_enum
                po_objs_list.append(solution.objectives._data)
                po_vars_list.append(po_vars)
            po_objs_arr = np.array(po_objs_list)
            po_confs_arr = np.array(po_vars_list)
            return po_objs_arr, po_confs_arr, model_infer_info

    def set_var_types(self, vars_ktype: list) -> bool:
        """
        :param var_types: list, variable types (float, integer, binary, enum)
        :return:
                flag_mixed_var_type: bool, to indicate whether the variable types
                are mixed (True) or all the same (False)
        """
        # n_vars = len(var_types)
        n_vars = len(vars_ktype)
        # find list of indices for different variable types
        float_inds = [i for i, x in enumerate(vars_ktype) if x == VarTypes.FLOAT]
        int_inds = [i for i, x in enumerate(vars_ktype) if x == VarTypes.INT]
        # bool variable class is extended from the interger variable class
        binary_inds = [i for i, x in enumerate(vars_ktype) if x == VarTypes.BOOL]
        # make it global as it will be used to decode the solutions
        # in the method self.solve
        global enum_inds
        enum_inds = [i for i, x in enumerate(vars_ktype) if x == VarTypes.CATEGORY]

        # set variable types for the problem
        if len(float_inds) > 0:
            for i in float_inds:
                if global_var_range[i][0] == global_var_range[i][1]:
                    self.problem.types[i] = Constant(global_var_range[i][0])
                else:
                    self.problem.types[i] = Real(
                        global_var_range[i][0], global_var_range[i][1]
                    )
        if len(int_inds) > 0:
            for i in int_inds:
                # print(f"ind is {i}")
                if global_var_range[i][0] == global_var_range[i][1]:
                    self.problem.types[i] = Constant(global_var_range[i][0])
                else:
                    self.problem.types[i] = Integer(
                        global_var_range[i][0], global_var_range[i][1]
                    )
        if len(binary_inds) > 0:
            for i in binary_inds:
                self.problem.types[i] = Integer(0, 1)
        if len(enum_inds) > 0:
            # Platypus does not support the categorical variable type
            # Here the ENUM variable type is transformed by using INTEGER type to:
            # 1) indicate the indices of categorical values
            # 2) will be decoded back to the categorical values
            # when calculating values of objectives and constraints,
            #   and return final variable values of Pareto solutions
            for i in enum_inds:
                self.problem.types[i] = Integer(0, len(global_var_range[i]) - 1)

        if len(float_inds) + len(int_inds) + len(binary_inds) + len(enum_inds) == 0:
            raise Exception(
                "ERROR: No feasilbe variables provided, please check the "
                "variable types setting!"
            )
        assert (
            len(float_inds) + len(int_inds) + len(binary_inds) + len(enum_inds)
            == n_vars
        )

        # check whether the variable types are mixed
        # if (len(float_inds) == n_vars) or (len(int_inds) == n_vars)
        # or (len(binary_inds) == n_vars) or (
        #         len(enum_inds) == n_vars):
        #     flag_mixed_var_type = False
        # else:
        #     flag_mixed_var_type = True
        flag_mixed_var_type = True
        return flag_mixed_var_type

    def problem_def(self, vars_list: List[Any]) -> List[Any]:
        """
        define the problem with objective and constraints, only
        support variables as input.
        :param vars:  list, variable types (float, integer, binary, enum)
        :return:
                f_list: list, values of objective functions
                g_list: list, values of constraint functions
        """
        # defined_functions need the input variables to be array
        # with shape([n, n_vars]), where n can be any positive number
        vars = np.array(vars_list)
        vars = vars.reshape([1, vars.shape[0]])

        if len(enum_inds) > 0:
            vars_decoded = np.ones_like(vars) * np.inf
            for i in enum_inds:
                ind = int(vars[0, i])
                decoded_enum = global_var_range[i][ind]
                vars_decoded[0, i] = decoded_enum
        else:
            vars_decoded = vars

        mesh_theta_c: Union[np.ndarray, th.Tensor]
        theta_p_s: Union[np.ndarray, th.Tensor]
        if self.use_ag:
            assert isinstance(self.non_decision_tabular_features, pd.DataFrame)
            mesh_theta_c = np.tile(
                vars_decoded[:, : self.len_theta_c], (self.n_stages, 1)
            )
            len_theta_p_s = self.len_theta_p + self.len_theta_s

            if self.is_query_control:
                assert vars_decoded.shape[1] == (
                    self.len_theta_c + self.len_theta_p + self.len_theta_s
                )
                theta_p_s = np.tile(
                    vars_decoded[:, self.len_theta_c :], (self.n_stages, 1)
                )
            else:
                assert vars_decoded.shape[1] == (
                    self.len_theta_c
                    + (self.len_theta_p + self.len_theta_s) * self.n_stages
                )
                theta_p_s = vars_decoded[:, self.len_theta_c :].reshape(
                    self.n_stages, len_theta_p_s
                )
            theta = np.concatenate([mesh_theta_c, theta_p_s], axis=1)
            start_pred = time.time()
            objs = self.obj_model(
                self.graph_embeddings.cpu().numpy(),
                self.non_decision_tabular_features,
                theta,
                self.ag_model,
            )
            n_evals = theta.shape[0]
            tc_pred = time.time() - start_pred
            model_infer_info.append([n_evals, tc_pred])

        else:
            # the formats of f_list(and g_list) is required as list[value1, value2, ...]
            assert isinstance(self.non_decision_tabular_features, th.Tensor)
            mesh_theta_c = th.tensor(vars_decoded[:, : self.len_theta_c]).repeat(
                self.n_stages, 1
            )
            len_theta_p_s = self.len_theta_p + self.len_theta_s
            if self.is_query_control:
                theta_p_s = th.tensor(vars_decoded[:, self.len_theta_c :]).repeat(
                    self.n_stages, 1
                )
            else:
                theta_p_s = th.tensor(vars_decoded[:, self.len_theta_c :]).reshape(
                    self.n_stages, len_theta_p_s
                )
            theta = th.cat([mesh_theta_c, theta_p_s], dim=1)
            objs = self.obj_model(
                self.graph_embeddings,
                self.non_decision_tabular_features,
                theta.type(th.float32),
                "",
            )
        query_objs = objs.sum(0).tolist()
        f_list = [query_objs[0], query_objs[1]]

        return f_list
