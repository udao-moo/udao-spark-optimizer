# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Directed Acyclic Graph (DAG) optimization methods
#
# Created at 18/04/2024

import itertools
import sys
from typing import Any, List, Tuple, Union

import numpy as np
import torch as th
from udao.optimization.utils.moo_utils import is_pareto_efficient

from udao_spark.optimizer.utils import timeis


class DAGOpt:
    def __init__(
        self,
        obj_values_all_subQs: np.ndarray,
        config_all_subQs: Union[th.Tensor, np.ndarray],
        indices_all_subQs: np.ndarray,
        algorithm: str,
        runmode: str,
    ) -> None:
        """
        initialize input for DAG optimization methods, which recover the subQ-level
        solutions into the query-level Pareto optimal solutions
        :param obj_values_all_subQs: subQ-level solutions that are Pareto optimal under
                each theta_c
        :param config_all_subQs: the corresponding configurations for all subQs
        :param indices_all_subQs: the indices of subQs, theta_c and optimal theta_p for
                the obj_valeus_all_subQs, with shape (#total_solution, 3),
                e.g. np.array([[0, 1, 2]]) indicates the second solution of subQ0 under
                theta_c_1 (indexing from 0).
        :param algorithm: the DAG optimization method, including:
                - GD: General Divide-and-conquer;
                - WS&int: Weighted Sum (WS)-based method, int is an integer showing the

                        number of weights used (e.g. 11);
                - B: Boundary-based method
        :param runmode: to indicate whether using multiprocessing
        """
        self.obj_values_all_subQs = obj_values_all_subQs
        self.config_all_subQs = config_all_subQs
        self.indices_all_subQs = indices_all_subQs
        self.algorithm = algorithm
        self.runmode = runmode

        self.len_theta = config_all_subQs.shape[1]
        self.n_objs = obj_values_all_subQs.shape[1]
        self.verbose = False

        all_function_names = [
            method
            for method in DAGOpt.__dict__
            if callable(getattr(DAGOpt, method)) and not method.startswith("__")
        ]
        self.global_time_cost: dict
        self.global_time_cost = {k: [] for k in all_function_names}

    def solve(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        THe entry of the DAG Optimization methods, which returns query-level Pareto
        optimal solutions recovered from the subQ-level Pareto optimal solutions.
        :return: query_obj_values: shape (#solutions, n_objs), here n_objs = 2, with
                                    latency and cost
                query_configurations: configurations of all subQs, shape (#solutions
                                    , len_one_theta * n_subQs)
        """
        if "WS" in self.algorithm:
            # the number of weights
            n_ws = int(self.algorithm.split("&")[1])
            ws_steps = 1 / (int(n_ws) - 1)
            w1 = np.hstack([np.arange(0, 1, ws_steps), 1])
            w2 = 1 - w1
            ws_pairs = [[w1, w2] for w1, w2 in zip(w1, w2)]

            (query_obj_values, query_configurations), tc_ws_based = self._ws_based(
                self.obj_values_all_subQs,
                self.config_all_subQs,
                self.indices_all_subQs,
                ws_pairs,
            )
            self.global_time_cost[self._ws_based.__name__].append(tc_ws_based)
        elif self.algorithm == "GD":
            (
                query_obj_values,
                query_configurations,
            ), tc_general_div_and_conq = self._general_div_and_conq(
                self.obj_values_all_subQs,
                self.config_all_subQs,
                self.indices_all_subQs,
            )
            self.global_time_cost[self._general_div_and_conq.__name__].append(
                tc_general_div_and_conq
            )
        # elif self.algorithm == "B":
        elif "B" in self.algorithm:
            (
                query_obj_values,
                query_configurations,
            ), tc_boundary_based = self._boundary_based(
                self.obj_values_all_subQs, self.config_all_subQs, self.indices_all_subQs
            )
            self.global_time_cost[self._boundary_based.__name__].append(
                tc_boundary_based
            )
        else:
            raise Exception(
                f"mode {self.algorithm} is not supported in" f" {classmethod}!"
            )

        return query_obj_values, query_configurations, self.global_time_cost

    @timeis
    def _boundary_based(
        self,
        obj_values_all_subQs: np.ndarray,
        config_all_subQs: Union[th.Tensor, np.ndarray],
        indices_all_subQs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Boundary-based methods, which get the two boundary subQ-level solutions
        under each theta_c, with minimum latency and cost, and then recover the
        query-level Pareto optimal solutions from the boundary solutions.
        :param obj_values_all_subQs: the subQ-level Pareto optimal solutions, which
         are Pareto optimal under each theta_c and subQ, shape (#total_solutions,
          n_objs), here n_objs=2 with latency and cost.
        :param config_all_subQs: the corresponding configurations, shape
        (#total_solutions, len_one_theta), here
                len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
        :param indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
        for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                 e.g. np.array([[0, 1, 2]]) indicates the second solution of subQ0
                 under theta_c_1 (indexing from 0).
        :return: query_obj_values: shape (#solutions, n_objs), here n_objs = 2, with
                            latency and cost
                query_configurations: configurations of all subQs, shape
                            (#solutions, len_one_theta * n_subQs)
        """
        # fixme: double-check
        (
            sorted_obj_values_with_split_subQs,
            sorted_configs_with_split_subQs,
            sorted_subQ_indices,
        ), tc_get_solutions_with_sorted_subQ_ids = self.get_results_sorted_subQ_ids(
            obj_values_all_subQs,
            config_all_subQs,
            indices_all_subQs,
        )
        self.global_time_cost[self.get_results_sorted_subQ_ids.__name__].append(
            tc_get_solutions_with_sorted_subQ_ids
        )

        n_subQs = len(sorted_obj_values_with_split_subQs)
        subQs = list(range(n_subQs))

        (
            boundary_obj_values_all_subQs,
            boundary_configs_all_subQs,
        ), tc_compute_boundary_all_subQs = self.get_boundary_all_subQs(
            subQs,
            sorted_obj_values_with_split_subQs,
            sorted_configs_with_split_subQs,
            sorted_subQ_indices,
        )
        self.global_time_cost[self.get_boundary_all_subQs.__name__].append(
            tc_compute_boundary_all_subQs
        )

        n_theta_c = np.unique(sorted_subQ_indices[:, 1]).shape[0]
        n_objs = self.n_objs
        len_theta = self.len_theta
        (
            boundary_query_objs,
            boundary_configs_all_subQs_arr,
        ), tc_compute_query_solutions = self.compute_query_solutions(
            n_subQs,
            n_theta_c,
            n_objs,
            len_theta,
            boundary_obj_values_all_subQs,
            boundary_configs_all_subQs,
        )
        self.global_time_cost[self.compute_query_solutions.__name__].append(
            tc_compute_query_solutions
        )

        (
            pareto_query_obj_values,
            pareto_query_configs,
        ), tc_filter_dominated_boundary_based = self.filter_dominated_boundary_based(
            boundary_query_objs,
            boundary_configs_all_subQs_arr,
        )
        self.global_time_cost[self.filter_dominated_boundary_based.__name__].append(
            tc_filter_dominated_boundary_based
        )

        return pareto_query_obj_values, pareto_query_configs

    @timeis
    def _ws_based(
        self,
        obj_values_all_subQs: np.ndarray,
        configs_all_subQs: Union[th.Tensor, np.ndarray],
        indices_all_subQs: np.ndarray,
        ws_pairs: List[List[Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted-Sum (WS)-based Approximation: using WS in each subQ to select the
        optimal subQ-level solution for each theta_c, and then compose them to get
        the query-level solutions.
        :param obj_values_all_subQs: the subQ-level Pareto optimal solutions, which
            are Pareto optimal under each theta_c and subQ, shape (#total_solutions,
            n_objs), here n_objs=2 with latency and cost.
        :param config_all_subQs: the corresponding configurations, shape
                (#total_solutions, len_one_theta), here
                len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
        :param indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
        for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                            e.g. np.array([[0, 1, 2]]) indicates the second solution
                             of subQ0 under theta_c_1 (indexing from 0).
        :param ws_pairs: weight pairs, e.g. [[0.1, 0.9], [0.9, 0.1]].
        :return: query_obj_values: shape (#solutions, n_objs), here n_objs = 2, with
                            latency and cost
                query_configurations: configurations of all subQs, shape (#solutions,
                            len_one_theta * n_subQs)
        """
        # fixme: to double-check
        (
            sorted_obj_values_all_subQs,
            sorted_configs_all_subQs,
            sorted_indices,
        ), tc_sort_input = self.process_input(
            obj_values_all_subQs, configs_all_subQs, indices_all_subQs
        )
        self.global_time_cost[self.process_input.__name__].append(tc_sort_input)

        n_subQs = len(sorted_obj_values_all_subQs)
        subQs = list(range(n_subQs))
        (query_obj_values, query_configs), tc_ws_all_subQs = self.ws_all_subQs(
            subQs,
            sorted_obj_values_all_subQs,
            sorted_configs_all_subQs,
            ws_pairs,
        )
        self.global_time_cost[self.ws_all_subQs.__name__].append(tc_ws_all_subQs)
        (
            pareto_query_obj_values,
            pareto_query_configs,
        ), tc_filter_dominated_ws_based = self.filter_dominated_ws_based(
            query_obj_values, query_configs
        )
        self.global_time_cost[self.filter_dominated_ws_based.__name__].append(
            tc_filter_dominated_ws_based
        )

        return pareto_query_obj_values, pareto_query_configs

    @timeis
    def _general_div_and_conq(
        self,
        obj_values_all_subQs: np.ndarray,
        configs_all_subQs: Union[th.Tensor, np.ndarray],
        indices_all_subQs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        General Divide-and-conquer approach: following the divide-and-conquer
        framework, iteratively merge two subQs of the DAG structure, where merging
        enumerates and filter dominated solutions among all combinations of solutions
        in two subQs with the constraint of identical theta_c
        :param obj_values_all_subQs: the subQ-level Pareto optimal solutions, which
                are Pareto optimal under each theta_c and subQ, shape
                (#total_solutions, n_objs), here n_objs=2 with latency and cost.
        :param config_all_subQs: the corresponding configurations, shape
                (#total_solutions, len_one_theta), here
                len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
        :param indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
                for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                e.g. np.array([[0, 1, 2]]) indicates the second solution of subQ0
                        under theta_c_1 (indexing from 0).
        :return: query_obj_values: shape (#solutions, n_objs), here n_objs = 2,
                                    with latency and cost
                query_configurations: configurations of all subQs, shape
                                    (#solutions, len_one_theta * n_subQs)
        """
        # fixme: to double-check
        (
            sorted_obj_values_all_subQs,
            sorted_configs_all_subQs,
            sorted_indices,
        ), tc_process_input = self.process_input(
            obj_values_all_subQs, configs_all_subQs, indices_all_subQs
        )
        self.global_time_cost[self.process_input.__name__].append(tc_process_input)

        n_subQs = len(sorted_obj_values_all_subQs)
        subQs = list(range(n_subQs))
        (
            query_obj_values_list,
            query_configs_list,
        ), tc_traverse_all_subQs = self.traverse_all_subQs_non_recur(
            subQs, sorted_obj_values_all_subQs, sorted_configs_all_subQs
        )
        self.global_time_cost[self.traverse_all_subQs_non_recur.__name__].append(
            tc_traverse_all_subQs
        )

        (
            pareto_query_obj_values,
            pareto_query_configs,
        ), tc_filter_dominated_general_div_and_conq = self.filter_query_dominated_GD(
            query_obj_values_list,
            query_configs_list,
        )
        self.global_time_cost[self.filter_query_dominated_GD.__name__].append(
            tc_filter_dominated_general_div_and_conq
        )

        return pareto_query_obj_values, pareto_query_configs

    @timeis
    def process_input(
        self,
        obj_values_all_subQs: np.ndarray,
        configs_all_subQs: Union[th.Tensor, np.ndarray],
        indices_all_subQs: np.ndarray,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], np.ndarray]:
        """
        Process the input for DAG optimization, to get the splitted solutions with
        the same theta_c and subQ. as the order of subQs and theta_c of current
        solutions is hard to locate
        :param obj_values_all_subQs: the subQ-level Pareto optimal solutions, which
                are Pareto optimal under each theta_c and subQ, shape
                (#total_solutions, n_objs), here n_objs=2 with latency and cost.
        :param configs_all_subQs: the corresponding configurations, shape
                (#total_solutions, len_one_theta), here
                len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
        :param indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
                for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                 e.g. np.array([[0, 1, 2]]) indicates the second solution of subQ0
                 under theta_c_1 (indexing from 0).
        :return: sorted_obj_values_all_subQs_all_theta_c: sorted objective values
                by following the increasing order of subQs, theta_c and theta_p
                indices,
                NOTE:
                - it follows the subQ order, and each element is a list representing
                 a solution set of a subQ;
                - each subQ solution set is a list, follows the order of theta_c ids,
                 where each element is a solution array of one specific theta_c with
                  shape (n_optimal_theta_p, n_objs).
                - Similar in sorted_configs_all_subQs_all_theta_c.
                sorted_configs_all_subQs_all_theta_c: sorted configurations by
                             following the increasing order of subQs, theta_c and
                             theta_p indices
                sorted_subQs_indices: indices with increasing order of subQs,
                            theta_c and theta_p indices
        """
        # fixme: to double-check whether it is necessary
        (
            sorted_obj_values_with_split_subQs,
            sorted_configs_with_split_subQs,
            sorted_subQs_indices,
        ), tc_get_solutions_with_sorted_subQ_ids = self.get_results_sorted_subQ_ids(
            obj_values_all_subQs,
            configs_all_subQs,
            indices_all_subQs,
        )
        self.global_time_cost[self.get_results_sorted_subQ_ids.__name__].append(
            tc_get_solutions_with_sorted_subQ_ids
        )

        (
            sorted_obj_values_all_subQs_all_theta_c,
            sorted_configs_all_subQs_all_theta_c,
        ), tc_split_subQs_set_with_theta_c = self.split_subQs_set_with_theta_c(
            sorted_obj_values_with_split_subQs,
            sorted_configs_with_split_subQs,
            sorted_subQs_indices,
        )
        self.global_time_cost[self.split_subQs_set_with_theta_c.__name__].append(
            tc_split_subQs_set_with_theta_c
        )

        return (
            sorted_obj_values_all_subQs_all_theta_c,
            sorted_configs_all_subQs_all_theta_c,
            sorted_subQs_indices,
        )

    @timeis
    def get_results_sorted_subQ_ids(
        self,
        obj_values_all_subQs: np.ndarray,
        config_all_subQs: Union[th.Tensor, np.ndarray],
        indices_all_subQs: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Re-order the solutions with the increasing order of subQs, theta_c and
        theta_p indices
        :param obj_values_all_subQs: the subQ-level Pareto optimal solutions,
                which are Pareto optimal under each theta_c and subQ, shape
                 (#total_solutions, n_objs), here n_objs=2 with latency and cost.
        :param configs_all_subQs: the corresponding configurations, shape
                (#total_solutions, len_one_theta), here
                len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
        :param indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
                for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                            e.g. np.array([[0, 1, 2]]) indicates the second solution
                             of subQ0 under theta_c_1 (indexing from 0).
        :return: sorted_obj_values_all_subQs: sorted objective values by following
                the increasing order of subQs, theta_c and theta_p indices,
                    NOTE:
                    - it follows the subQ order, and each element is an array
                    representing a solution set of a subQ;
                    - Similar in sorted_configs_all_subQs.
                sorted_configs_all_subQs: sorted configurations by following the
                    increasing order of subQs, theta_c and theta_p indices
                sorted_subQs_indices: indices with increasing order of subQs,
                    theta_c and theta_p indices
        """
        # fixme: to double-check whether it is necessary
        # example
        # >> > a = [1, 5, 1, 4, 3, 4, 4]  # First column
        # >> > b = [9, 4, 0, 4, 0, 2, 1]  # Second column
        # >> > ind = np.lexsort((b, a))  # Sort by a, then by b
        # sort the indices first by subQ_ids, then theta_c ids
        # and finally theta_p ids
        sorted_index = np.lexsort(
            (indices_all_subQs[:, 2], indices_all_subQs[:, 1], indices_all_subQs[:, 0])
        )
        sorted_subQs_indices = indices_all_subQs[sorted_index]
        subQs = np.unique(sorted_subQs_indices[:, 0])

        if isinstance(config_all_subQs, np.ndarray):
            sorted_configs = config_all_subQs[sorted_index]
        else:
            assert isinstance(config_all_subQs, th.Tensor)
            sorted_configs = config_all_subQs.numpy()[sorted_index]

        sorted_obj_values = obj_values_all_subQs[sorted_index]

        sorted_obj_values_all_subQ = [
            sorted_obj_values[np.where(sorted_subQs_indices[:, 0] == i)] for i in subQs
        ]
        sorted_configs_all_subQ = [
            sorted_configs[np.where(sorted_subQs_indices[:, 0] == i)] for i in subQs
        ]

        return sorted_obj_values_all_subQ, sorted_configs_all_subQ, sorted_subQs_indices

    @timeis
    def split_subQs_set_with_theta_c(
        self,
        sorted_obj_values_all_subQ: List[np.ndarray],
        sorted_configs_all_subQ: List[np.ndarray],
        sorted_subQ_indices: np.ndarray,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        split the subQ-level solution array into the solution sets of different theta_c
        :param sorted_obj_values_all_subQ: sorted objective values by following the
                increasing order of subQs, theta_c and theta_p indices,
                    NOTE:
                    - it follows the subQ order, and each element is an array
                    representing a solution set of a subQ;
                    - Similar in sorted_configs_all_subQs.
        :param sorted_configs_all_subQ: sorted configurations by following the
                increasing order of subQs, theta_c and theta_p indices.
        :param sorted_subQ_indices: indices with increasing order of subQs, theta_c
                and theta_p indices
        :return: sorted_obj_values_all_subQs_all_theta_c: sorted objective values
                        by following the increasing order of subQs, theta_c and
                        theta_p indices,
                        NOTE:
                        - it follows the subQ order, and each element is a list
                        representing a solution set of a subQ;
                        - each subQ solution set is a list, follows the order of
                        theta_c ids, where each element is a
                        solution array of one specific theta_c with shape
                        (n_optimal_theta_p, n_objs).
                        - Similar in sorted_configs_all_subQs_all_theta_c.
                sorted_configs_all_subQs_all_theta_c: sorted configurations by
                            following the increasing order of subQs, theta_c and
                            theta_p indices
        """
        # fixme: to double-check
        subQs = list(range(len(sorted_obj_values_all_subQ)))
        n_theta_c = np.unique(sorted_subQ_indices[:, 1]).shape[0]
        print(f"The number of theta_c in DAG Optimization is: {n_theta_c}")

        sorted_obj_values_all_subQ_all_theta_c = []
        sorted_configs_all_subQ_all_theta_c = []
        for subQ_id, subQ_obj_values, subQ_configs in zip(
            subQs, sorted_obj_values_all_subQ, sorted_configs_all_subQ
        ):
            # Note: np.unique will sort the result
            # fixme:
            len_opt_theta_p_counts = np.unique(
                sorted_subQ_indices[np.where(sorted_subQ_indices[:, 0] == subQ_id)][
                    :, 1
                ],
                return_counts=True,
            )[1]
            # Examples
            # --------
            # >> > a = np.array([[1, 2, 3], [4, 5, 6]])
            # >> > a
            # array([[1, 2, 3],
            #        [4, 5, 6]])
            # >> > np.cumsum(a)
            # array([1, 3, 6, 10, 15, 21])
            cumsum_counts = np.cumsum(len_opt_theta_p_counts)[:-1]
            obj_values_list = np.split(subQ_obj_values, cumsum_counts)
            configs_list = np.split(subQ_configs, cumsum_counts)
            assert len(obj_values_list) == n_theta_c
            assert len(configs_list) == n_theta_c

            sorted_obj_values_all_subQ_all_theta_c.append(obj_values_list)
            sorted_configs_all_subQ_all_theta_c.append(configs_list)

        return (
            sorted_obj_values_all_subQ_all_theta_c,
            sorted_configs_all_subQ_all_theta_c,
        )

    ############ sub-functions under in General Divide-and-Conquer ########
    @timeis
    def traverse_all_subQs_non_recur(
        self,
        subQs: List[int],
        obj_values_all_subQs: List[List[np.ndarray]],
        configs_all_subQs: List[List[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Traverse all the subQs and merge them iteratively to recover the query-level
         solutions from the subQ-level.
        :param subQs: subQ indices
        :param obj_values_all_subQs: sorted objective values by following the
                increasing order of subQs, theta_c and theta_p indices,
                NOTE:
                - it follows the subQ order, and each element is a list representing
                 a solution set of a subQ;
                - each subQ solution set is a list, follows the order of theta_c ids,
                 where each element is a solution array of one specific theta_c with
                  shape (n_optimal_theta_p, n_objs).
                - Similar in configs_all_subQs.
        :param configs_all_subQs: sorted configurations by following the increasing
                order of subQs, theta_c and theta_p indices
        :return: pareto_query_obj_values: Pareto query-level solutions of all theta_c,
                        the list follows the order of theta_c indices, and each array
                        represents the query-level Pareto optimal solution of each
                        theta_c, with shape (n_solutions, n_objs).
                pareto_query_configs: corresponding configurations of
                        pareto_query_obj_values, where the list follows the order of
                        theta_c indices, and each array represents the query-level
                        configuration of each theta_c,
                        with shape (n_solutions, len_one_theta * n_subQs)
        """

        subQs_str = [f"subQ{subQ_id}" for subQ_id in subQs]
        stack_obj_values = obj_values_all_subQs.copy()
        stack_configs = configs_all_subQs.copy()

        pareto_query_obj_values: List[np.ndarray]
        pareto_query_configs: List[np.ndarray]
        while len(subQs_str) > 1:
            if len(subQs_str) == 2:
                subQ1_id = subQs_str[0]
                subQ2_id = subQs_str[1]

                (
                    pareto_query_obj_values,
                    pareto_merged_configs,
                ), tc_merged_two_subQs = self.merge_two_subQs(
                    stack_obj_values,
                    stack_configs,
                    sub_problem_index=0,
                )
                self.global_time_cost[self.merge_two_subQs.__name__].append(
                    tc_merged_two_subQs
                )

                subQs_str.append(f"{subQ1_id}+{subQ2_id}")
                subQs_str.remove(subQ1_id)
                subQs_str.remove(subQ2_id)
                assert len(subQs_str) == 1

                (
                    merged_subQs_indices_order,
                    tc_get_merged_subQs_order,
                ) = self.get_merged_subQs_order(subQ1_id, subQ2_id)
                self.global_time_cost[self.get_merged_subQs_order.__name__].append(
                    tc_get_merged_subQs_order
                )

                (
                    pareto_query_configs,
                    tc_reorder_query_config,
                ) = self.reorder_subQs_for_configs(
                    merged_subQs_indices_order, pareto_merged_configs
                )
                self.global_time_cost[self.reorder_subQs_for_configs.__name__].append(
                    tc_reorder_query_config
                )

                assert len(pareto_query_obj_values) == len(pareto_query_configs)
                assert all(
                    [
                        x.shape[0] == y.shape[0]
                        for x, y in zip(pareto_query_obj_values, pareto_query_configs)
                    ]
                )
                assert all(
                    [
                        x.shape[1] == len(merged_subQs_indices_order) * self.len_theta
                        for x in pareto_query_configs
                    ]
                )
                break

            else:
                (
                    updated_stack_obj_values,
                    updated_stack_configs,
                ), tc_merge_more_than_two_subQs = self.solve_more_than_two_subQs(
                    subQs_str,
                    stack_obj_values,
                    stack_configs,
                )
                self.global_time_cost[self.solve_more_than_two_subQs.__name__].append(
                    tc_merge_more_than_two_subQs
                )
                stack_obj_values = updated_stack_obj_values.copy()
                stack_configs = updated_stack_configs.copy()
                assert len(stack_obj_values) == len(stack_configs)

        return pareto_query_obj_values, pareto_query_configs

    @timeis
    def compute_merged_values(
        self,
        subQ1: Tuple[List[np.ndarray], List[np.ndarray]],
        subQ2: Tuple[List[np.ndarray], List[np.ndarray]],
    ) -> Tuple[List[List[Any]], List[List[Tuple[Any, ...]]]]:
        """fixme: to double-check
        Merge the two subQs into one, and compute the merged "subQ-level" values.
        e.g. Merged latency/cost of two subQs is the sum of the latency/cost of the
        two subQs.
        :param subQ1: includes objective values (subQ1[0]) and configurations
                    (subQ1[1]) of subQ1
                    - subQ1[0]: list follows the order of theta_c, each array is the
                    solution set with shape (n_optimal_theta_p, n_objs)
                    - subQ1[1]: list follows the order of theta_c, each array is the
                    confiuration set with shape (n_optimal_theta_p,
                    len_one_theta/len_one_theta * n_merged_subQs),
                        where n_merged_subQs represents the merged subQs,
                        e.g. 2 means the configuration set is already merged from
                        two subQs.
                    - similar in subQ2.
        :param subQ2: includes objective values (subQ2[0]) and configurations
                    (subQ2[1]) of subQ2
        :return:
        merged_obj_values_all_theta_c: merged objective values of two subQs under
                all theta_c, where each element in the list is an 2d list following
                 the order of theta_c indices.
        merged_configs_all_theta_c: merged solution optimal choices of two subQs
                under all theta_c, where each element in the list of tuples follows
                 the order of theta_c indices.
                    e.g. given two theta_c, subQ1 has 2 optimal solutions, and
                    subQ2 has 2 optimal solutions.
                    [[(0, 0), (1, 0)],  # two optimal merged solutions under
                                            theta_c1, i.e. (subQ1_0, subQ2_0) and
                                        # (subQ1_0, subQ2_0), where 0/1 represents
                                            the row indices of solution set
                                        # array in subQ1[0] and subQ2[0]
                     [(0, 0)]] # one optimal merged solution under theta_c2
        """
        merged_obj_values_all_theta_c = []
        merged_configs_all_theta_c = []
        for i, (subQ1_obj_values_theta_ci, subQ2_obj_values_theta_ci) in enumerate(
            zip(subQ1[0], subQ2[0])
        ):
            all_indices_choices = list(
                itertools.product(
                    *[
                        list(range(subQ1_obj_values_theta_ci.shape[0])),
                        list(range(subQ2_obj_values_theta_ci.shape[0])),
                    ]
                )
            )
            obj_values_list, tc_enumeration = self.enumeration(
                all_indices_choices,
                subQ1_obj_values_theta_ci,
                subQ2_obj_values_theta_ci,
            )
            self.global_time_cost[self.enumeration.__name__].append(tc_enumeration)
            merged_obj_values_all_theta_c.append(obj_values_list)
            merged_configs_all_theta_c.append(all_indices_choices)
        return merged_obj_values_all_theta_c, merged_configs_all_theta_c

    @timeis
    def enumeration(
        self,
        all_indices_choices: List[Tuple[Any, ...]],
        subQ1_obj_values: np.ndarray,
        subQ2_obj_values: np.ndarray,
    ) -> List[List[Any]]:
        """fixme: to double-check
        enumerate all combination of solutions in two subQs and compute the merged
        values.
        e.g. under theta_c1, subQ1 has [[latency1, cost1], [latency2, cost2]],
                subQ2 has [[latency1', cost1']]
        - merged values of subQ1 and subQ2, e.g. latency1+latency1', cost1+cost1'
        - all combinations: [[latency1+latency1', cost1+cost1'], [latency2+latency1',
            cost2+cost1']]
        :param all_indices_choices: the indices choices of two subQs
                e.g. under theta_c1, subQ1 has [[latency1, cost1], [latency2, cost2]]
                , subQ2 has [[latency1', cost1']]
                combination choices of subQ1 and subQ2 is [(0, 0), (1, 0)],
                where (1, 0) indicates subQ1 chooses the second solution (i.e.
                [latency2, cost2]),
                and subQ2 chooses the first solution (i.e. [latency1', cost1'])
        :param subQ1_obj_values: objective values of all theta_c in subQ1, follows
                the order of theta_c indices, each element is an array of one
                specific theta_c with shape (n_optimal_theta_p, n_objs)
        :param subQ2_obj_values: objective values of all theta_c in subQ2, follows
                the order of theta_c indices, each element is an array of one specific
                 theta_c with shape (n_optimal_theta_p, n_objs)
        :return: merged objective values: follows the order of the theta_c indices,
                each element is a set of merged objective values,
                 e.g. [latency1+latency1', cost1+cost1']
        """
        obj_values_list = []
        for subQ1_choice, subQ2_choice in all_indices_choices:
            latency, cost = np.sum(
                [subQ1_obj_values[subQ1_choice], subQ2_obj_values[subQ2_choice]],
                axis=0,
                dtype=np.float64,
            )
            obj_values_list.append([latency, cost])
        return obj_values_list

    @timeis
    def filter_dominated_merged_values(
        self,
        merged_obj_values_all_theta_c: List[List[Any]],
        merged_configs_indices_all_theta_c: List[List[Tuple[Any, ...]]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """fixme: to double-check
        filter the dominated among merged solutions for each theta_c
        :param merged_obj_values_all_theta_c: list follows the order of the theta_c
                indices, each element is a set of merged objective values,
                e.g. [latency1+latency1', cost1+cost1']
        :param merged_configs_indices_all_theta_c: the solution choices in subQ1 and
                subQ2.
                e.g. [[(0, 0), (1, 0)], [(0, 0)]] for two theta_c.
        :return: pareto_merged_obj_values: keep the non-dominated solution for each
                    solution set of each theta_c, it follows the order of theta_c,
                    each element is an array of merged objective values.
                pareto_merged_configs_indices: keep the non-dominated configurations
                    for each solution set of each theta_c, it follows the order of
                    theta_c, each element is an array of merged configurations, with
                     shape (n_optimal_solutions, 2),
                     where colume1: solution choice of subQ1,
                        colume2: choice of subQ2.
        """
        pareto_merged_obj_values = []
        pareto_merged_configs_indices = []
        for obj_values, configs in zip(
            merged_obj_values_all_theta_c, merged_configs_indices_all_theta_c
        ):
            obj_values_arr = np.array(obj_values)
            pareto_indices = is_pareto_efficient(obj_values_arr).tolist()
            pareto_obj_values = obj_values_arr[pareto_indices]
            pareto_configs_indices = np.array(configs)[pareto_indices]

            pareto_merged_obj_values.append(pareto_obj_values)
            pareto_merged_configs_indices.append(pareto_configs_indices)

        return pareto_merged_obj_values, pareto_merged_configs_indices

    @timeis
    def merge_two_subQs(
        self,
        stack_obj_values: List[List[np.ndarray]],
        stack_configs: List[List[np.ndarray]],
        sub_problem_index: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Merge two subQs with merged Pareto optimal objective values and
        configurations.
        :param stack_obj_values: the objective values of subQs not merged yet.
                        It follows the order of subQs, and each element is a list
                        of solution arrays under all theta_c.
        :param stack_configs: the configurations of subQs not merged yet.
                        It follows the order of subQs, and each element is a list
                        of configuration arrays under all theta_c
        :param sub_problem_index: the index of the sub-problems.
                        e.g. given a DAG with 5 subQs, it splits into 3 sub-problems
                         where each has at most 2 subQs.
                            i.e. sub-problem1: [subQ1, subQ2],
                            sub-problem2: [subQ3, subQ4], sub-problem3: [subQ5]
        :return: pareto_merged_obj_values: keep the non-dominated solution for each
                        solution set of each theta_c, it follows the order of theta_c
                        , each element is an array of merged objective values.
                pareto_merged_configs: keep the non-dominated configurations for
                        each configuration set of each theta_c, it follows the order
                        of theta_c, each element is an array of merged configurations,
                        with shape (n_optimal_solutions,
                        n_merged_subQs * len_one_theta).
                        - where n_merged_subQs represents the merged subQs,
                        e.g. 2 means the configuration set is already merged from
                        two subQs.
        """

        subQ1 = (
            stack_obj_values[sub_problem_index * 2],
            stack_configs[sub_problem_index * 2],
        )
        subQ2 = (
            stack_obj_values[sub_problem_index * 2 + 1],
            stack_configs[sub_problem_index * 2 + 1],
        )

        (
            merged_obj_values_all_theta_c,
            merged_configs_indices_all_theta_c,
        ), tc_compute_merged_values = self.compute_merged_values(subQ1, subQ2)
        self.global_time_cost[self.compute_merged_values.__name__].append(
            tc_compute_merged_values
        )

        (
            pareto_merged_obj_values,
            pareto_merged_configs_indices,
        ), tc_filter_dominated_merged_values = self.filter_dominated_merged_values(
            merged_obj_values_all_theta_c,
            merged_configs_indices_all_theta_c,
        )
        self.global_time_cost[self.filter_dominated_merged_values.__name__].append(
            tc_filter_dominated_merged_values
        )

        (
            pareto_merged_configs,
            tc_update_merged_configs,
        ) = self.update_merged_configs_with_indices_choices(
            pareto_merged_configs_indices,
            subQ1[1],
            subQ2[1],
        )
        self.global_time_cost[
            self.update_merged_configs_with_indices_choices.__name__
        ].append(tc_update_merged_configs)

        return pareto_merged_obj_values, pareto_merged_configs

    @timeis
    def update_merged_configs_with_indices_choices(
        self,
        merged_configs_indices_all_theta_c: List[np.ndarray],
        subQ1_configs: List[np.ndarray],
        subQ2_configs: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        recover the configuration features from the choices of each subQ.
        :param merged_configs_indices_all_theta_c: the non-dominated configurations
                for each solution set of each theta_c, it follows the order of
                theta_c, each element is an array of merged configurations, with
                shape (n_optimal_solutions, 2),
                where colume1: solution choice of subQ1,
                        colume2: choice of subQ2.
        :param subQ1_configs: configurations in subQ1, follows the order of theta_c
                indices, each element is an configuration array under one theta_c
        :param subQ2_configs: configurations in subQ2, follows the order of theta_c
                indices, each element is an configuration array under one theta_c
        :return: the configuration features for each configuration set of each
                        theta_c, it follows the order of theta_c, each element is
                        an array of merged configurations, with shape
                        (n_optimal_solutions, n_merged_subQs * len_one_theta).
                        - where n_merged_subQs represents the merged subQs,
                        e.g. 2 means the configuration set is already merged from
                        two subQs.
        """
        merged_configs_all_theta_c = []
        for i, indices_choices_theta_ci in enumerate(
            merged_configs_indices_all_theta_c
        ):
            # each choice is under each theta_c
            subQ1_configs_to_merge = subQ1_configs[i][indices_choices_theta_ci[:, 0]]
            subQ2_configs_to_merge = subQ2_configs[i][indices_choices_theta_ci[:, 1]]
            merged_configs_theta_ci = np.hstack(
                (subQ1_configs_to_merge, subQ2_configs_to_merge)
            )
            assert merged_configs_theta_ci.shape[0] == indices_choices_theta_ci.shape[0]
            merged_configs_all_theta_c.append(merged_configs_theta_ci)

        return merged_configs_all_theta_c

    @timeis
    def solve_more_than_two_subQs(
        self,
        subQs_str: List[str],
        stack_obj_values: List[List[np.ndarray]],
        stack_configs: List[List[np.ndarray]],
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        merge multiple subQs.
        e.g. given a DAG with 5 subQs, it splits into 3 sub-problems where each has
        at most 2 subQs.
        i.e. sub-problem1: [subQ1, subQ2], sub-problem2: [subQ3, subQ4],
        sub-problem3: [subQ5]
        :param subQs_str: subQ indices,
                    e.g. initial: ["subQ1", "subQ2", "subQ3", "subQ4", "subQ5"]
                         after looping all sub-problems: ["subQ5", "subQ1+subQ2",
                         "subQ3+subQ4"]
        :param stack_obj_values: the objective values of subQs not merged yet.
                        It follows the order of subQs, and each element is a list of
                         solution arrays under all theta_c.
        :param stack_configs: the configurations of subQs not merged yet.
                        It follows the order of subQs, and each element is a list of
                         configuration arrays under all theta_c.
        :return: pareto_merged_obj_values: follows subQ orders, keep the non-dominated
                        solution for each solution set
                        of each theta_c, it follows the order of theta_c, each element
                        is an array of merged objective values.
                pareto_merged_configs: follows subQ orders, keep the non-dominated
                        configurations for each configuration set of each theta_c,
                        each element is a list which follows the order of theta_c,
                        this list includes arrays of merged configurations,
                        with shape (n_optimal_solutions, n_merged_subQs *
                        len_one_theta).
                        - where n_merged_subQs represents the merged subQs,
                        e.g. 2 means the configuration set is already merged from
                        two subQs.
        """
        n_sub_problems = (
            int(len(subQs_str) / 2)
            if len(subQs_str) % 2 == 0
            else int(len(subQs_str) / 2) + 1
        )
        current_subQs_str = subQs_str.copy()
        new_stack_obj_values: List[List[np.ndarray]] = []
        new_stack_configs: List[List[np.ndarray]] = []
        for i in range(n_sub_problems):
            if i * 2 + 1 == len(current_subQs_str):
                new_stack_obj_values.insert(0, stack_obj_values[-1])
                new_stack_configs.insert(0, stack_configs[-1])
            else:
                subQ1_id = current_subQs_str[i * 2]
                subQ2_id = current_subQs_str[i * 2 + 1]

                (
                    pareto_merged_obj_values,
                    pareto_merged_configs,
                ), tc_merge_two_subQs = self.merge_two_subQs(
                    stack_obj_values,
                    stack_configs,
                    sub_problem_index=i,
                )
                self.global_time_cost[self.merge_two_subQs.__name__].append(
                    tc_merge_two_subQs
                )

                new_stack_obj_values.append(pareto_merged_obj_values)
                new_stack_configs.append(pareto_merged_configs)

                subQs_str.append(f"{subQ1_id}+{subQ2_id}")
                subQs_str.remove(subQ1_id)
                subQs_str.remove(subQ2_id)

        return new_stack_obj_values, new_stack_configs

    @timeis
    def get_merged_subQs_order(
        self,
        subQ1_id: str,
        subQ2_id: str,
    ) -> List[int]:
        """
        in the final merging operation, get the order of merged subQs.
        :param subQ1_id: subQ ids, e.g. subQ1_id = "subQ2"
        :param subQ2_id: subQ2_id = "subQ0+subQ1"
        :return: the order of merged subQs, e.g. [2, 0, 1]
        """
        if "+" in subQ1_id:
            subQ1_indices_order = [int(x.split("subQ")[1]) for x in subQ1_id.split("+")]
        else:
            subQ1_indices_order = [int(subQ1_id.split("subQ")[1])]

        if "+" in subQ2_id:
            subQ2_indices_order = [int(x.split("subQ")[1]) for x in subQ2_id.split("+")]
        else:
            subQ2_indices_order = [int(subQ2_id.split("subQ")[1])]

        subQs_indices_order = subQ1_indices_order + subQ2_indices_order

        return subQs_indices_order

    @timeis
    def reorder_subQs_for_configs(
        self,
        merged_subQs_indices_order: List[int],
        pareto_merged_configs: List[np.ndarray],
    ) -> List[np.ndarray]:
        """fixme: double-check
        re-order the FINAL configurations to follow subQ0, subQ1, ..., subQm.
        :param merged_subQs_indices_order: the merged_subQ_indices,
                e.g. ["subQ2", "subQ0+subQ1"]
        :param pareto_merged_configs: follows subQ orders, keep the non-dominated
                configurations for each configuration set of each theta_c, it
                follows the order of theta_c, each element is an array of merged
                configurations, with shape (n_optimal_solutions,
                n_subQs * len_one_theta). following the order of merged subQs,
                e.g. ["subQ2", "subQ0+subQ1"]
        :return: pareto_query_configs: the configurations of all theta_c, where
                        each element is any array with shape (n_optimal_solutions,
                         n_subQs * len_one_theta), following the order subQ0,
                         subQ1,...
        """
        n_subQs = len(merged_subQs_indices_order)
        split_configs = [np.hsplit(x, n_subQs) for x in pareto_merged_configs]
        sort_subQ_indices = np.argsort(merged_subQs_indices_order)
        # let configurations follow the order of subQ0, 1, ...
        pareto_query_configs = [
            np.concatenate([y[x] for x in sort_subQ_indices], axis=1)
            for y in split_configs
        ]
        return pareto_query_configs

    @timeis
    def filter_query_dominated_GD(
        self,
        query_obj_values_list: List[np.ndarray],
        query_configs_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        filter the query-level dominated solutions among all theta_c
        :param query_obj_values_list: query-level pareto objective values under each
                theta_c, follows the order of theta_c
        :param query_configs_list: query-level pareto configurations under each
                theta_c, follows the order of theta_c
        :return: uniq_pareto_query_obj_values: unique query-level pareto optimal
                        objective values, shape (n_solutions, n_objs)
                uniq_pareto_query_configs: unique query-level pareto optimal
                        configurations, shape (n_solutions, n_subQs * len_one_theta)
        """
        query_obj_values_arr = np.concatenate(query_obj_values_list)
        query_configs_arr = np.concatenate(query_configs_list)
        pareto_query_indices = is_pareto_efficient(query_obj_values_arr)
        pareto_query_obj_values = query_obj_values_arr[pareto_query_indices]
        paretp_query_configs = query_configs_arr[pareto_query_indices]
        uniq_pareto_query_obj_values, uniq_pareto_query_indices = np.unique(
            pareto_query_obj_values, axis=0, return_index=True
        )
        uniq_pareto_query_configs = paretp_query_configs[uniq_pareto_query_indices]
        print(f"Query-level Pareto frontier is {uniq_pareto_query_obj_values}")
        return uniq_pareto_query_obj_values, uniq_pareto_query_configs

    ############ sub-functions under in WS-Based Approximation ########
    @timeis
    def ws_per_subQ(
        self,
        ws_pairs: List[List[float]],
        obj_values_one_subQ: List[np.ndarray],
        configs_one_subQ: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applying Weighted Sum (WS) in each subQ under all theta_c
        :param ws_pairs: weight pairs, e.g. [[0.1, 0.9], [0.9, 0.1]].
        :param obj_values_one_subQ: the subQ-level Pareto optimal objective values of
                ONE subQ, which are Pareto optimal under each theta_c,
                shape (#total_solutions, n_objs), here n_objs=2 with latency and
                cost.
        :param configs_one_subQ: the subQ-level Pareto optimal configurations of ONE
                subQ, which are Pareto optimal under each theta_c, shape
                (#total_solutions, len_one_theta).
        :return: obj_values_obj_subQ_select:  the selected objective values of this
                        subQ, with shape (n_theta_c, n_ws * n_objs),
                        e.g. given weights [[0.1, 0.9], [0.9, 0.1]]
                                  #####################################
                        e.g.      #    [0.1, 0.9]   #    [0.9, 0.1]   #
                        theta_c1  # latency1, cost1 # latency2, cost2 #
                        theta_c2  # latency1',cost1'# latency2',cost2'#
                                  #####################################
                configs_one_subQ_select: the selected configurations of this subQ,
                        with shape (n_theta_c, n_ws * len_one_theta)
        """
        # the objective values of all theta_c,
        # a list with n_theta_c items, each item is an array with shape
        # (n_solutions, n_objs)
        # obj_values_one_subQ = obj_values_all_subQs[int(subQ_id)]
        # configs_one_subQ = configs_all_subQs[int(subQ_id)]
        n_theta_c = len(obj_values_one_subQ)
        assert n_theta_c == len(configs_one_subQ)

        # under each theta_c, apply WS for all solutions within the node
        # 1. normalize objs
        # 2. norm_objs * ws_pairs_arr = shape (n_objs * n_solutions) \times
        #                               shape (n_ws * n_objs)
        #    = shape (n_solutions * n_ws)
        # 3. find the index of the minimum weighted sum values in step 2 under all
        # weights:
        #    output --> shape (1, n_ws)

        # considering multiple theta_c, where under each theta_c, n_solutions could
        # differ
        # --> find the n_max, and pad the solutions array of the node to shape
        # (n_theta_c, n_max)
        # output shape should be (n_theta_c, n_ws)

        # the maximum number of solutions among all theta_c (n_max)
        n_max_solutions = max([x.shape[0] for x in obj_values_one_subQ])

        (
            obj_values_one_subQ_pad_list,
            configs_one_subQ_pad_list,
        ), tc_pad = self.pad_solutions_one_subQ(
            n_max_solutions,
            obj_values_one_subQ,
            configs_one_subQ,
        )
        self.global_time_cost[self.pad_solutions_one_subQ.__name__].append(tc_pad)

        obj_values_one_subQ_pad_arr = np.array(obj_values_one_subQ_pad_list)
        configs_one_subQ_pad_arr = np.array(configs_one_subQ_pad_list)
        # double-check whether the paded objective values and configurations have
        # the same #rows
        assert obj_values_one_subQ_pad_arr.shape[0] == n_theta_c * n_max_solutions
        assert configs_one_subQ_pad_arr.shape[0] == obj_values_one_subQ_pad_arr.shape[0]

        ####### 1. normalize obj_values per theta_c #######
        ## find the min and max of objectives
        objs_min = np.array([x.min(0) for x in obj_values_one_subQ])
        objs_max = np.array([x.max(0) for x in obj_values_one_subQ])
        assert np.all(objs_max - objs_min >= 0)

        ## pad them with n_max_solutions among all theta_c
        objs_min_repeat = np.repeat(objs_min, n_max_solutions, axis=0)
        objs_max_repeat = np.repeat(objs_max, n_max_solutions, axis=0)
        assert objs_max_repeat.shape[0] == n_theta_c * n_max_solutions
        assert objs_max_repeat.shape[0] == objs_min_repeat.shape[0]

        norm_obj_values = (obj_values_one_subQ_pad_arr - objs_min_repeat) / (
            objs_max_repeat - objs_min_repeat
        )
        # the case with only one solution under one theta_c
        inds_one_solution = np.where(~(objs_max_repeat - objs_min_repeat).any(axis=1))[
            0
        ]
        # set them to be 0 as it won't affect the ws values
        norm_obj_values[inds_one_solution] = 0
        ######## 2. norm_objs * ws_pairs_arr #######
        # shape (n_ws, 2)
        ws_pairs_arr = np.array(ws_pairs)
        # shape (n_theta_c*n_max, n_ws)
        ws_obj_values = np.matmul(norm_obj_values, ws_pairs_arr.T)

        ######## 3. find the index of min_ws, shape (n_theta_c, n_ws) #######
        split_ws_obj_values = np.split(ws_obj_values, n_theta_c)
        inds_select = [np.argmin(x, axis=0) for x in split_ws_obj_values]

        # return selected solution among all weights (n_theta_c, n_ws * n_objs)
        obj_values_obj_subQ_select = np.vstack(
            [
                np.hstack([x[ind] for ind in inds_all_ws])
                for x, inds_all_ws in zip(obj_values_one_subQ, inds_select)
            ]
        )
        configs_one_subQ_select = np.vstack(
            [
                np.hstack([x[ind] for ind in inds_all_ws])
                for x, inds_all_ws in zip(configs_one_subQ, inds_select)
            ]
        )

        return obj_values_obj_subQ_select, configs_one_subQ_select

    @timeis
    def pad_solutions_one_subQ(
        self,
        n_max_solutions: int,
        obj_values_one_subQ: List[np.ndarray],
        configs_one_subQ: List[np.ndarray],
    ) -> Tuple[List[List[Any]], List[List[Any]]]:
        """fixme: to double-check
        Given each theta_c has different number of optimal solutions, the function
        pads the solutions under the same theta_c into the same number of rows, so
        that the solution set of all theta_c is represented as one array.
        The aim is to apply WS once for solutions among all theta_c.
        :param n_max_solutions: the number of maximum number of solutions among all
                theta_c
        :param obj_values_one_subQ: the objective values set of each subQ, list:
                where each item is the solution array (shape: (n_solutions, n_objs))
                 of each theta_c
        :param configs_one_subQ: the configuration set of each subQ, list: where
                each item is the solution array
                (shape: (n_solutions, n_objs)) of each theta_c
        :return: obj_values_one_subQ_pad_list: follows the order of theta_c, length
                        is n_theta_c * n_max_solutions objective values.
                        e.g. given two theta_c in subQ1,
                        - theta_c1: [[25, 20], [23, 23]], theta_c2: [[23, 25]]
                        after padding. [[25, 20], [23, 23], [23, 25],
                        [9223372036854775807, 9223372036854775806]]
                configs_one_subQ_pad_list: follows the order of theta_c, length is
                        n_theta_c * n_max_solutions configurations.
        """
        ## pad the solution array with n_max by the value -1
        n_diff_to_n_max = [n_max_solutions - x.shape[0] for x in obj_values_one_subQ]
        obj_values_one_subQ_pad_list = []
        configs_one_subQ_pad_list = []
        for obj_values, configs, n_diff in zip(
            obj_values_one_subQ, configs_one_subQ, n_diff_to_n_max
        ):
            # set the extension with two different and large values,
            # to make sure: 1) max-min != 0 in normalization, if a theta_c only has
            #                   one optimal theta_p;
            #               2) won't be selected after ws (to minimize)
            obj_values_one_subQ_pad_list.extend(
                obj_values.tolist() + [[sys.maxsize, sys.maxsize - 1]] * n_diff
            )
            configs_one_subQ_pad_list.extend(
                configs.tolist() + [[-1] * self.len_theta] * n_diff
            )

        return obj_values_one_subQ_pad_list, configs_one_subQ_pad_list

    @timeis
    def ws_all_subQs(
        self,
        subQs: List[int],
        obj_values_all_subQs: List[List[np.ndarray]],
        configs_all_subQs: List[List[np.ndarray]],
        ws_pairs: List[List[float]],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply WS for all subQs
        :param subQs: subQ ids
        :param obj_values_all_subQs: sorted objective values by following the
                increasing order of subQs, theta_c and theta_p indices,
                NOTE:
                - it follows the subQ order, and each element is a list representing
                 a solution set of a subQ;
                - each subQ solution set is a list, follows the order of theta_c ids,
                 where each element is a solution array of one specific theta_c with
                  shape (n_optimal_theta_p, n_objs).
                - Similar in configs_all_subQs.
        :param configs_all_subQs: sorted configurations by following the increasing
                order of subQs, theta_c and theta_p indices
        :param ws_pairs: weight pairs, e.g. [[0.1, 0.9], [0.9, 0.1]].
        :return: query_obj_values: query objective values array with shape
                        (n_ws * n_theta_c, n_objs) objective values.
                        e.g. given two theta_c among all subQs, where
                        weight pairs = [[0.1, 0.9], [0.9, 0.1]]
                        - subQ1: [[np.array([[25, 20],[23, 23]]),np.array([[23, 25]])],
                        - subQ2: [np.array([[22, 24]]),np.array([[21, 26]])],
                        - subQ3: [np.array([[27, 27],[28, 21]]),np.array([[25, 29]])]
                        after WS selection. np.array([[75, 65], [69, 80],
                        [72, 74], [69, 80]])
                query_configs: configurations of all subQs, list follows the order
                        of theta_c, each element in the list is an array with shape
                        (n_ws * n_theta_c, len_one_theta)
        """
        select_obj_values_list = []
        select_configs_list = []
        for subQ_id in subQs:
            obj_values_one_subQ = obj_values_all_subQs[int(subQ_id)]
            configs_one_subQ = configs_all_subQs[int(subQ_id)]

            (
                select_obj_values_one_subQ,
                select_configs_one_subQ,
            ), tc_ws_per_subQ = self.ws_per_subQ(
                ws_pairs, obj_values_one_subQ, configs_one_subQ
            )
            self.global_time_cost[self.ws_per_subQ.__name__].append(tc_ws_per_subQ)

            select_obj_values_list.append(select_obj_values_one_subQ)
            select_configs_list.append(select_configs_one_subQ)
        # sum all nodes: shape (n_theta_c, n_ws * n_objs),
        # where 0 (even id) is lat, 1 (odd id) is cost
        query_values_all_ws = np.sum(np.array(select_obj_values_list), axis=0)
        # query_obj_values: shape (n_ws * n_theta_c, n_objs)
        # order: under each weight, following n_theta_c solutions,
        # i.e. n_theta_c_ws_1 solutions, ..., n_theta_c_ws_i,...
        query_obj_values = np.vstack(
            np.split(query_values_all_ws, len(ws_pairs), axis=1)
        )

        # query_confs_all_ws: a list with length of n_stages
        # each item in the list is an array with shape (n_ws * n_theta_c,
        # len_one_theta)
        # order of each array: under each weight, following n_theta_c solutions
        query_configs = [
            np.vstack(np.split(x, len(ws_pairs), axis=1)) for x in select_configs_list
        ]

        return query_obj_values, query_configs

    @timeis
    def filter_dominated_ws_based(
        self,
        query_obj_values: np.ndarray,
        query_configs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter query-level dominated solutions among all theta_c
        :param query_obj_values: query objective values array with shape
                (n_ws * n_theta_c, n_objs) objective values.
                e.g. given two theta_c among all subQs, where weight pairs =
                    [[0.1, 0.9], [0.9, 0.1]]
                - subQ1: [[np.array([[25, 20], [23, 23]]),np.array([[23, 25]])],
                - subQ2: [np.array([[22, 24]]),np.array([[21, 26]])],
                - subQ3: [np.array([[27, 27], [28, 21]]), np.array([[25, 29]])]
                after WS selection. np.array([[75, 65], [69, 80], [72, 74],
                [69, 80]])
        :param query_configs: configurations of all subQs, list follows the order
                of theta_c, each element in the list is an array with shape
                (n_ws * n_theta_c, len_one_theta)
        :return: uniq_pareto_query_obj_values: unique query-level pareto optimal
                        objective values, shape (n_solutions, n_objs)
                uniq_pareto_query_configs: unique query-level pareto optimal
                        configurations, shape (n_solutions, n_subQs * len_one_theta)
        """
        pareto_query_ind = is_pareto_efficient(query_obj_values)
        pareto_query_obj_values = query_obj_values[pareto_query_ind]

        # shape n_solutions, n_subQs * len_one_theta)
        # given m subQs, the columes orders are :
        # 0, ..., len_one_theta, ... , (m-1) * len_one_theta
        pareto_query_configs = np.hstack([x[pareto_query_ind] for x in query_configs])

        uniq_pareto_query_obj_values, uniq_pareto_query_inds = np.unique(
            pareto_query_obj_values, axis=0, return_index=True
        )
        uniq_pareto_query_configs = pareto_query_configs[uniq_pareto_query_inds]
        print(f"Query-level Pareto frontier is {uniq_pareto_query_obj_values}")
        return uniq_pareto_query_obj_values, uniq_pareto_query_configs

    ########## sub-functions under in Boundary-based Approximation ########
    @timeis
    def get_boundary_all_subQs(
        self,
        subQs: List[int],
        obj_values_all_subQs: List[np.ndarray],
        configs_all_subQs: List[np.ndarray],
        sorted_subQ_indices: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # fixme: to double-check
        """
        get the boundary subQ-level solutions of each theta_c
        :param subQs: subQ ids
        :param obj_values_all_subQs: sorted objective values by following the
                increasing order of subQs, theta_c and theta_p indices,
                NOTE:
                - it follows the subQ order, and each element is an array
                representing a solution set of a subQ;
                - Similar in configs_all_subQs.
        :param configs_all_subQs: sorted configurations by following the increasing
                order of subQs, theta_c and theta_p indices
        :param sorted_subQ_indices: indices with increasing order of subQs, theta_c
                and theta_p indices
        :return: boundary_obj_values_all_subQs: boundary objective values with
                        minimum latency/cost among all theta_c;
                        list: length n_subQs * n_theta_c
                boundary_configs_all_subQs: boundary configurations with minimum
                        latency/cost among all theta_c;
                        list: length n_subQs * n_theta_c
        """

        sorted_obj_values_all_subQs = np.concatenate(obj_values_all_subQs)
        sorted_configs_all_subQs = np.concatenate(configs_all_subQs)

        n_theta_c = np.unique(sorted_subQ_indices[:, 1]).shape[0]
        n_subQs = len(subQs)

        # Find the indices where the theta_c id values change
        theta_c_ids = sorted_subQ_indices[:, 1]
        change_indices = np.where(np.diff(theta_c_ids) != 0)[0] + 1

        # Define starting and ending indices for each sub-array
        # b/w each start and end, it is the sub-array under the same subQ and theta_c id
        # e.g. given two subQs and two theta_c, sub-arrays follow the order:
        # [subQ0, theta_c0], [subQ0, theta_c1], [subQ1, theta_c0], [subQ1, theta_c1]
        starts = np.concatenate([np.array([0]), change_indices])
        ends = np.concatenate([starts[1:], [theta_c_ids.shape[0]]])
        assert starts.shape[0] == ends.shape[0]
        assert starts.shape[0] == n_theta_c * n_subQs

        # find the indices of solutions with minimum latency and cost for each sub-array
        sorted_sub_arrays_ids = [
            [
                np.argmin(sorted_obj_values_all_subQs[start:end][:, 0]),
                np.argmin(sorted_obj_values_all_subQs[start:end][:, 1]),
            ]
            for start, end in zip(starts, ends)
        ]

        # the list length is n_subQ * n_theta_c
        # the order is:
        #   [subQ0, theta_c0],
        #   [subQ0, theta_c1],
        #   [subQ1, theta_c0],
        #   [subQ1, theta_c1]
        # each item is an array with shape (n_objs, n_objs), here n_objs = 2
        # the first row is latency, and the second is cost
        obj_values_boundary = [
            np.vstack(
                [
                    sorted_obj_values_all_subQs[start:end][boundary_inds[0]],
                    sorted_obj_values_all_subQs[start:end][boundary_inds[1]],
                ]
            )
            for start, end, boundary_inds in zip(starts, ends, sorted_sub_arrays_ids)
        ]
        configs_boundary = [
            np.vstack(
                [
                    sorted_configs_all_subQs[start:end][boundary_inds[0]],
                    sorted_configs_all_subQs[start:end][boundary_inds[1]],
                ]
            )
            for start, end, boundary_inds in zip(starts, ends, sorted_sub_arrays_ids)
        ]

        assert len(obj_values_boundary) == len(configs_boundary)

        return obj_values_boundary, configs_boundary

    @timeis
    def compute_query_solutions(
        self,
        n_subQs: int,
        n_theta_c: int,
        n_objs: int,
        len_theta: int,
        boundary_obj_values_all_subQs: List[np.ndarray],
        boundary_configs_all_subQs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute the query-level values from the boundary subQ-level solutions.
        :param n_subQs: the number of subQs.
        :param n_theta_c: the number of theta_c.
        :param n_objs: the number of objectives.
        :param len_theta: the number of parameters for one subQ.
        :param boundary_obj_values_all_subQs: boundary objective values with minimum
                latency/cost among all theta_c;
                list: length n_subQs * n_theta_c
        :param boundary_configs_all_subQs: boundary configurations with minimum
                latency/cost among all theta_c;
                list: length n_subQs * n_theta_c, follows the order of theta_c ids
        :return: boundary_query_objs: query-level objective values with shape
                        (2 * n_theta_c, n_objs)
                boundary_configs_all_subQs_arr: configurations of all subQs,
                            with shape (2 * n_theta_c, n_subQs * len_one_theta)
        """
        # fixme: double-check
        boundary_obj_values_all_subQs_arr = np.concatenate(
            np.split(np.array(boundary_obj_values_all_subQs), n_subQs), axis=2
        ).reshape((n_theta_c * n_objs, n_subQs * n_objs))
        boundary_configs_all_subQs_arr = np.concatenate(
            np.split(np.array(boundary_configs_all_subQs), n_subQs), axis=2
        ).reshape((n_theta_c * n_objs, n_subQs * len_theta))

        query_lat = np.sum(boundary_obj_values_all_subQs_arr[:, 0::2], axis=1)
        query_cost = np.sum(boundary_obj_values_all_subQs_arr[:, 1::2], axis=1)
        boundary_query_objs = np.vstack((query_lat, query_cost)).T

        return boundary_query_objs, boundary_configs_all_subQs_arr

    @timeis
    def filter_dominated_boundary_based(
        self,
        boundary_query_objs: np.ndarray,
        boundary_configs_all_subQs_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter the query-level dominated solutions among all theta_c.
        :param boundary_query_objs: query-level objective values with shape
                (2 * n_theta_c, n_objs)
        :param boundary_configs_all_subQs_arr: configurations of all subQs,
                            with shape (2 * n_theta_c, n_subQs * len_one_theta)
        :return: uniq_pareto_query_objs: unique query-level pareto optimal objective
                        values, shape (n_solutions, n_objs)
                uniq_pareto_query_configs: unique query-level pareto optimal
                        configurations, shape (n_solutions, n_subQs * len_one_theta)
        """
        pareto_query_indices = is_pareto_efficient(boundary_query_objs)
        pareto_query_objs = boundary_query_objs[pareto_query_indices]
        pareto_query_configs = boundary_configs_all_subQs_arr[pareto_query_indices]
        uniq_pareto_query_objs, uniq_pareto_query_inds = np.unique(
            pareto_query_objs, axis=0, return_index=True
        )
        uniq_pareto_query_configs = pareto_query_configs[uniq_pareto_query_inds]

        return uniq_pareto_query_objs, uniq_pareto_query_configs
