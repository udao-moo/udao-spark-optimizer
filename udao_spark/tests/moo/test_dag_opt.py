# Copyright (c) 2020 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: test functions used in DAG optimization
#
# Created at 24/04/2024

import itertools

import numpy as np

from udao_spark.optimizer.moo_algos.dag_optimization import DAGOpt

sorted_indices_all_subQs = np.array(
    [
        [0, 1, 0],
        [0, 1, 1],
        [0, 8, 0],
        [1, 1, 0],
        [1, 8, 0],
        [2, 1, 0],
        [2, 1, 1],
        [2, 8, 0],
    ]
)
sorted_obj_values_all_subQs = [
    [np.array([[25, 20], [23, 23]]), np.array([[23, 25]])],
    [np.array([[22, 24]]), np.array([[21, 26]])],
    [np.array([[27, 27], [28, 21]]), np.array([[25, 29]])],
]

sorted_configs_all_subQs = [
    [
        np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
        np.array([[12, 16, 37, 25, 33, 1, 2]]),
    ],
    [np.array([[9, 4, 26, 27, 34, 1, 2]]), np.array([[12, 16, 37, 25, 33, 1, 2]])],
    [
        np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
        np.array([[12, 16, 26, 27, 34, 1, 2]]),
    ],
]
n_subQs = len(sorted_obj_values_all_subQs)
subQs = list(range(n_subQs))


class TestDAGOpt:
    def test_process_input(self, dag_opt: DAGOpt) -> None:
        dag_opt.algorithm = "GD"
        objective_values = dag_opt.obj_values_all_subQs
        configurations = dag_opt.config_all_subQs
        indices = dag_opt.indices_all_subQs

        (
            result_sorted_objective_values,
            result_sorted_configs,
            result_sorted_indices,
        ), _ = dag_opt.process_input(objective_values, configurations, indices)

        assert np.all(result_sorted_indices == sorted_indices_all_subQs)
        assert all(
            [
                [
                    np.all(result == expect)
                    for result, expect in zip(result_subQ, expect_subQ)
                ]
                for result_subQ, expect_subQ in zip(
                    result_sorted_objective_values, sorted_obj_values_all_subQs
                )
            ]
        )
        assert all(
            [
                [
                    np.all(result == expect)
                    for result, expect in zip(result_subQ, expect_subQ)
                ]
                for result_subQ, expect_subQ in zip(
                    result_sorted_configs, sorted_configs_all_subQs
                )
            ]
        )

    def test_get_solutions_with_sorted_subQ_ids(self, dag_opt: DAGOpt) -> None:
        dag_opt.algorithm = "GD"
        objective_values = dag_opt.obj_values_all_subQs
        configurations = dag_opt.config_all_subQs
        indices = dag_opt.indices_all_subQs

        (
            sorted_obj_values_with_split_subQs,
            sorted_configs_with_split_subQs,
            sorted_indices,
        ), _ = dag_opt.get_results_sorted_subQ_ids(
            objective_values, configurations, indices
        )

        expect_sorted_obj_values_with_split_subQs = [
            np.array([[25, 20], [23, 23], [23, 25]]),
            np.array([[22, 24], [21, 26]]),
            np.array([[27, 27], [28, 21], [25, 29]]),
        ]
        expect_sorted_configs_with_split_subQs = [
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                ]
            ),
            np.array([[9, 4, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 26, 27, 34, 1, 2],
                ]
            ),
        ]

        assert np.all(sorted_indices == sorted_indices_all_subQs)
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    sorted_obj_values_with_split_subQs,
                    expect_sorted_obj_values_with_split_subQs,
                )
            ]
        )
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    sorted_configs_with_split_subQs,
                    expect_sorted_configs_with_split_subQs,
                )
            ]
        )

    def test_split_subQs_set_with_theta_c(self, dag_opt: DAGOpt) -> None:
        sorted_obj_values_with_split_subQs = [
            np.array([[25, 20], [23, 23], [23, 25]]),
            np.array([[22, 24], [21, 26]]),
            np.array([[27, 27], [28, 21], [25, 29]]),
        ]
        sorted_configs_with_split_subQs = [
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                ]
            ),
            np.array([[9, 4, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 26, 27, 34, 1, 2],
                ]
            ),
        ]

        (
            sorted_obj_values_all_subQs_all_theta_c,
            sorted_configs_all_subQs_all_theta_c,
        ), _ = dag_opt.split_subQs_set_with_theta_c(
            sorted_obj_values_with_split_subQs,
            sorted_configs_with_split_subQs,
            sorted_indices_all_subQs,
        )

        assert all(
            [
                [
                    np.all(result == expect)
                    for result, expect in zip(result_subQ, expect_subQ)
                ]
                for result_subQ, expect_subQ in zip(
                    sorted_obj_values_all_subQs_all_theta_c, sorted_obj_values_all_subQs
                )
            ]
        )
        assert all(
            [
                [
                    np.all(result == expect)
                    for result, expect in zip(result_subQ, expect_subQ)
                ]
                for result_subQ, expect_subQ in zip(
                    sorted_configs_all_subQs_all_theta_c, sorted_configs_all_subQs
                )
            ]
        )

    ######  sub-functions in WS-based Approaximation method
    def test_pad_solutions_one_subQ(self, dag_opt: DAGOpt) -> None:
        # choose one subQ
        subQ_id = 0
        obj_values_one_subQ = sorted_obj_values_all_subQs[subQ_id]
        configs_one_subQ = sorted_configs_all_subQs[subQ_id]
        n_max_solutions = max([x.shape[0] for x in obj_values_one_subQ])

        (
            obj_values_one_subQ_pad_list,
            configs_one_subQ_pad_list,
        ), tc_pad = dag_opt.pad_solutions_one_subQ(
            n_max_solutions,
            obj_values_one_subQ,
            configs_one_subQ,
        )

        expect_obj_values_one_subQ_pad_list = [
            [25, 20],
            [23, 23],
            [23, 25],
            [9223372036854775807, 9223372036854775806],
        ]
        expect_configs_one_subQ_pad_list = [
            [9, 4, 26, 27, 34, 1, 2],
            [9, 4, 37, 25, 33, 1, 2],
            [12, 16, 37, 25, 33, 1, 2],
            [-1, -1, -1, -1, -1, -1, -1],
        ]

        assert np.all(
            np.array(
                obj_values_one_subQ_pad_list == expect_obj_values_one_subQ_pad_list
            )
        )
        assert np.all(
            np.array(configs_one_subQ_pad_list == expect_configs_one_subQ_pad_list)
        )

    def test_ws_per_subQ(self, dag_opt: DAGOpt) -> None:
        # choose one subQ
        subQ_id = 0
        obj_values_one_subQ = sorted_obj_values_all_subQs[subQ_id]
        configs_one_subQ = sorted_configs_all_subQs[subQ_id]

        ws_pairs = [[0.1, 0.9], [0.9, 0.1]]

        # return selected solution among all weights (n_theta_c, n_ws * n_objs)
        (select_obj_values_arr, select_configs_arr), _ = dag_opt.ws_per_subQ(
            ws_pairs,
            obj_values_one_subQ,
            configs_one_subQ,
        )

        expect_select_obj_values_arr = np.array([[25, 20, 23, 23], [23, 25, 23, 25]])
        expect_select_configs_arr = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2, 9, 4, 37, 25, 33, 1, 2],
                [12, 16, 37, 25, 33, 1, 2, 12, 16, 37, 25, 33, 1, 2],
            ]
        )

        assert np.all(select_obj_values_arr == expect_select_obj_values_arr)
        assert np.all(select_configs_arr == expect_select_configs_arr)

    def test_ws_all_subQs(self, dag_opt: DAGOpt) -> None:
        ws_pairs = [[0.1, 0.9], [0.9, 0.1]]

        (query_obj_values, query_configs), _ = dag_opt.ws_all_subQs(
            subQs,
            sorted_obj_values_all_subQs,
            sorted_configs_all_subQs,
            ws_pairs,
        )

        expect_query_obj_values = np.array([[75, 65], [69, 80], [72, 74], [69, 80]])
        expect_query_configs = [
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                ]
            ),
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                ]
            ),
            np.array(
                [
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 26, 27, 34, 1, 2],
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 26, 27, 34, 1, 2],
                ]
            ),
        ]

        assert np.all(query_obj_values == expect_query_obj_values)
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(query_configs, expect_query_configs)
            ]
        )

    def test_filter_dominated_ws_based(self, dag_opt: DAGOpt) -> None:
        query_obj_values = np.array([[75, 65], [69, 80], [72, 74], [69, 80]])
        query_configs = [
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                ]
            ),
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                ]
            ),
            np.array(
                [
                    [9, 4, 37, 25, 33, 1, 2],
                    [12, 16, 26, 27, 34, 1, 2],
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 26, 27, 34, 1, 2],
                ]
            ),
        ]

        (
            pareto_query_obj_values,
            pareto_query_configs,
        ), _ = dag_opt.filter_dominated_ws_based(query_obj_values, query_configs)

        expect_pareto_query_obj_values = np.array([[69, 80], [72, 74], [75, 65]])
        expect_pareto_query_configs = np.array(
            [
                [
                    12,
                    16,
                    37,
                    25,
                    33,
                    1,
                    2,
                    12,
                    16,
                    37,
                    25,
                    33,
                    1,
                    2,
                    12,
                    16,
                    26,
                    27,
                    34,
                    1,
                    2,
                ],
                [
                    9,
                    4,
                    37,
                    25,
                    33,
                    1,
                    2,
                    9,
                    4,
                    26,
                    27,
                    34,
                    1,
                    2,
                    9,
                    4,
                    26,
                    27,
                    34,
                    1,
                    2,
                ],
                [
                    9,
                    4,
                    26,
                    27,
                    34,
                    1,
                    2,
                    9,
                    4,
                    26,
                    27,
                    34,
                    1,
                    2,
                    9,
                    4,
                    37,
                    25,
                    33,
                    1,
                    2,
                ],
            ]
        )

        assert np.all(pareto_query_obj_values == expect_pareto_query_obj_values)
        assert np.all(pareto_query_configs == expect_pareto_query_configs)

    ######  sub-functions in General Divide-and-Conquer method
    def test_enumeration(self, dag_opt: DAGOpt) -> None:
        subQ1_id = 0
        subQ2_id = 1
        theta_c_id = 0
        subQ1_obj_values_theta_c1 = sorted_obj_values_all_subQs[subQ1_id][theta_c_id]
        subQ2_obj_values_theta_c1 = sorted_obj_values_all_subQs[subQ2_id][theta_c_id]
        all_indices_choices = list(
            itertools.product(
                *[
                    list(range(subQ1_obj_values_theta_c1.shape[0])),
                    list(range(subQ2_obj_values_theta_c1.shape[0])),
                ]
            )
        )
        obj_values_list, _ = dag_opt.enumeration(
            all_indices_choices,
            np.array(subQ1_obj_values_theta_c1),
            np.array(subQ2_obj_values_theta_c1),
        )
        expect_obj_values_list = [[47.0, 44.0], [45.0, 47.0]]
        assert np.all(np.array(obj_values_list) == np.array(expect_obj_values_list))

    def test_compute_merged_values(self, dag_opt: DAGOpt) -> None:
        subQ1_id = 0
        subQ2_id = 1
        subQ1 = (
            sorted_obj_values_all_subQs[subQ1_id],
            sorted_configs_all_subQs[subQ1_id],
        )
        subQ2 = (
            sorted_obj_values_all_subQs[subQ2_id],
            sorted_configs_all_subQs[subQ2_id],
        )

        (
            merged_obj_values_all_theta_c,
            merged_configs_indices_all_theta_c,
        ), _ = dag_opt.compute_merged_values(
            subQ1,
            subQ2,
        )
        expect_merged_obj_values_all_theta_c = [
            [[47.0, 44.0], [45.0, 47.0]],
            [[44.0, 51.0]],
        ]
        expect_merged_configs_indices_all_theta_c = [[(0, 0), (1, 0)], [(0, 0)]]

        assert np.all(
            [
                np.all(np.array(result) == np.array(expect))
                for result, expect in zip(
                    merged_obj_values_all_theta_c,
                    expect_merged_obj_values_all_theta_c,
                )
            ]
        )
        assert np.all(
            [
                np.all(np.array(result) == np.array(expect))
                for result, expect in zip(
                    merged_configs_indices_all_theta_c,
                    expect_merged_configs_indices_all_theta_c,
                )
            ]
        )

    def test_filter_dominated_merged_values(self, dag_opt: DAGOpt) -> None:
        # NOTE: not the same as the expect_merged_obj_values_all_theta_c in
        # test_compute_merged_values
        merged_obj_values_all_theta_c = [[[47.0, 44.0], [45.0, 43.0]], [[44.0, 51.0]]]
        merged_configs_indices_all_theta_c = [[(0, 0), (1, 0)], [(0, 0)]]

        (
            pareto_merged_obj_values,
            pareto_merged_configs_indices,
        ), _ = dag_opt.filter_dominated_merged_values(
            merged_obj_values_all_theta_c, merged_configs_indices_all_theta_c
        )

        expect_pareto_merged_obj_values = [
            np.array([[45.0, 43.0]]),
            np.array([[44.0, 51.0]]),
        ]
        expect_pareto_merged_configs_indices = [np.array([[1, 0]]), np.array([[0, 0]])]

        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    pareto_merged_obj_values, expect_pareto_merged_obj_values
                )
            ]
        )
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    pareto_merged_configs_indices, expect_pareto_merged_configs_indices
                )
            ]
        )

    def test_update_merged_configs_with_indices_choices(self, dag_opt: DAGOpt) -> None:
        subQ1_id = 0
        subQ2_id = 1
        subQ1_configs = sorted_configs_all_subQs[subQ1_id]
        subQ2_configs = sorted_configs_all_subQs[subQ2_id]
        pareto_merged_configs_indices_choices = [np.array([[1, 0]]), np.array([[0, 0]])]

        (
            merged_configs_all_theta_c,
            _,
        ) = dag_opt.update_merged_configs_with_indices_choices(
            pareto_merged_configs_indices_choices,
            subQ1_configs,
            subQ2_configs,
        )

        expect_merged_configs_all_theta_c = [
            np.array([[9, 4, 37, 25, 33, 1, 2, 9, 4, 26, 27, 34, 1, 2]]),
            np.array([[12, 16, 37, 25, 33, 1, 2, 12, 16, 37, 25, 33, 1, 2]]),
        ]
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    merged_configs_all_theta_c, expect_merged_configs_all_theta_c
                )
            ]
        )

    def test_merge_two_subQs(self, dag_opt: DAGOpt) -> None:
        subQ1_id = 0
        subQ2_id = 1
        stack_obj_values = [
            sorted_obj_values_all_subQs[subQ1_id],
            sorted_obj_values_all_subQs[subQ2_id],
        ]
        stack_configs = [
            sorted_configs_all_subQs[subQ1_id],
            sorted_configs_all_subQs[subQ2_id],
        ]
        (pareto_merged_obj_values, pareto_merged_configs), _ = dag_opt.merge_two_subQs(
            stack_obj_values,
            stack_configs,
            sub_problem_index=0,
        )

        expect_pareto_merged_obj_values = [
            np.array([[47.0, 44.0], [45.0, 47.0]]),
            np.array([[44.0, 51.0]]),
        ]
        expect_pareto_merged_configs = [
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                ]
            ),
            np.array([[12, 16, 37, 25, 33, 1, 2, 12, 16, 37, 25, 33, 1, 2]]),
        ]
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    pareto_merged_obj_values, expect_pareto_merged_obj_values
                )
            ]
        )
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    pareto_merged_configs, expect_pareto_merged_configs
                )
            ]
        )

    def test_solve_more_than_two_subQs(self, dag_opt: DAGOpt) -> None:
        subQs_str = [f"subQ{subQ_id}" for subQ_id in subQs]
        stack_obj_values = sorted_obj_values_all_subQs.copy()
        stack_configs = sorted_configs_all_subQs.copy()
        (
            updated_stack_obj_values,
            updated_stack_configs,
        ), _ = dag_opt.solve_more_than_two_subQs(
            subQs_str,
            stack_obj_values,
            stack_configs,
        )

        expect_updated_stack_obj_values = [
            [np.array([[27, 27], [28, 21]]), np.array([[25, 29]])],
            [np.array([[47.0, 44.0], [45.0, 47.0]]), np.array([[44.0, 51.0]])],
        ]
        expect_updated_stack_configs = [
            [
                np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
                np.array([[12, 16, 26, 27, 34, 1, 2]]),
            ],
            [
                np.array(
                    [
                        [9, 4, 26, 27, 34, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                        [9, 4, 37, 25, 33, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                    ]
                ),
                np.array([[12, 16, 37, 25, 33, 1, 2, 12, 16, 37, 25, 33, 1, 2]]),
            ],
        ]

        assert all(
            [
                [
                    np.all(result == expect)
                    for result, expect in zip(result_subQ, expect_subQ)
                ]
                for result_subQ, expect_subQ in zip(
                    updated_stack_obj_values, expect_updated_stack_obj_values
                )
            ]
        )
        assert all(
            [
                [
                    np.all(result == expect)
                    for result, expect in zip(result_subQ, expect_subQ)
                ]
                for result_subQ, expect_subQ in zip(
                    updated_stack_configs, expect_updated_stack_configs
                )
            ]
        )

    def test_traverse_all_subQs_non_recur(self, dag_opt: DAGOpt) -> None:
        (
            pareto_query_obj_values,
            pareto_query_configs,
        ), _ = dag_opt.traverse_all_subQs_non_recur(
            subQs,
            sorted_obj_values_all_subQs,
            sorted_configs_all_subQs,
        )

        expect_pareto_query_obj_values = [
            np.array([[72.0, 74.0], [75.0, 65.0], [73.0, 68.0]]),
            np.array([[69.0, 80.0]]),
        ]
        expect_pareto_query_configs = [
            np.array(
                [
                    [
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                    ],
                    [
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                    ],
                    [
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        12,
                        16,
                        37,
                        25,
                        33,
                        1,
                        2,
                        12,
                        16,
                        37,
                        25,
                        33,
                        1,
                        2,
                        12,
                        16,
                        26,
                        27,
                        34,
                        1,
                        2,
                    ]
                ]
            ),
        ]

        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    pareto_query_obj_values, expect_pareto_query_obj_values
                )
            ]
        )
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    pareto_query_configs, expect_pareto_query_configs
                )
            ]
        )

    def test_get_merged_subQs_order(self, dag_opt: DAGOpt) -> None:
        subQ1_id = "subQ2"
        subQ2_id = "subQ0+subQ1"

        merged_subQs_ids_order, _ = dag_opt.get_merged_subQs_order(subQ1_id, subQ2_id)

        expect_merged_subQs_ids_order = [2, 0, 1]
        assert merged_subQs_ids_order == expect_merged_subQs_ids_order

    def test_reorder_subQs_for_query_configs(self, dag_opt: DAGOpt) -> None:
        # one theta_c, follows the order of subQ2, subQ0, subQ1.
        pareto_merged_configs = [np.array([[5, 6, 1, 2, 3, 4]])]
        merged_subQs_indices_order = [2, 0, 1]

        pareto_query_configs, _ = dag_opt.reorder_subQs_for_configs(
            merged_subQs_indices_order,
            pareto_merged_configs,
        )

        expect_pareto_query_configs = [np.array([1, 2, 3, 4, 5, 6])]
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    pareto_query_configs, expect_pareto_query_configs
                )
            ]
        )

    def test_filter_query_dominated_general_div_and_conq(self, dag_opt: DAGOpt) -> None:
        # NOTE: it is not the same as the expect_query_obj_values_list in
        # test_traverse_all_subQs_non_recur
        query_obj_values_list = [
            np.array([[72.0, 74.0], [75.0, 65.0], [71.0, 68.0]]),
            np.array([[69.0, 80.0]]),
        ]
        query_configs_list = [
            np.array(
                [
                    [
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                    ],
                    [
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                    ],
                    [
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                        9,
                        4,
                        26,
                        27,
                        34,
                        1,
                        2,
                        9,
                        4,
                        37,
                        25,
                        33,
                        1,
                        2,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        12,
                        16,
                        37,
                        25,
                        33,
                        1,
                        2,
                        12,
                        16,
                        37,
                        25,
                        33,
                        1,
                        2,
                        12,
                        16,
                        26,
                        27,
                        34,
                        1,
                        2,
                    ]
                ]
            ),
        ]

        (
            pareto_query_obj_values,
            pareto_query_configs,
        ), _ = dag_opt.filter_query_dominated_GD(
            query_obj_values_list,
            query_configs_list,
        )

        expect_pareto_query_obj_values = np.array(
            [[69.0, 80.0], [71.0, 68.0], [75.0, 65.0]]
        )
        expect_pareto_query_configs = np.array(
            [
                [
                    12,
                    16,
                    37,
                    25,
                    33,
                    1,
                    2,
                    12,
                    16,
                    37,
                    25,
                    33,
                    1,
                    2,
                    12,
                    16,
                    26,
                    27,
                    34,
                    1,
                    2,
                ],
                [
                    9,
                    4,
                    37,
                    25,
                    33,
                    1,
                    2,
                    9,
                    4,
                    26,
                    27,
                    34,
                    1,
                    2,
                    9,
                    4,
                    37,
                    25,
                    33,
                    1,
                    2,
                ],
                [
                    9,
                    4,
                    26,
                    27,
                    34,
                    1,
                    2,
                    9,
                    4,
                    26,
                    27,
                    34,
                    1,
                    2,
                    9,
                    4,
                    37,
                    25,
                    33,
                    1,
                    2,
                ],
            ]
        )
        assert np.all(pareto_query_configs == expect_pareto_query_configs)
        assert np.all(pareto_query_obj_values == expect_pareto_query_obj_values)

    ######  sub-functions in Boundary-based Approaximation method
    def test_compute_boundary_all_subQs(self, dag_opt: DAGOpt) -> None:
        sorted_obj_values_with_split_subQs = [
            np.array([[25, 20], [23, 23], [22, 24], [23, 25]]),
            np.array([[27, 27], [28, 21], [25, 26], [23, 29]]),
        ]
        sorted_configs_with_split_subQs = [
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [9, 4, 37, 25, 33, 1, 2],
                    [9, 4, 23, 24, 36, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                ]
            ),
            np.array(
                [
                    [9, 4, 26, 27, 34, 1, 2],
                    [12, 16, 37, 25, 33, 1, 2],
                    [12, 16, 32, 26, 31, 1, 2],
                    [12, 16, 26, 27, 34, 1, 2],
                ]
            ),
        ]
        sorted_indices_all_subQs = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 8, 0],
                [1, 1, 0],
                [1, 8, 0],
                [1, 8, 1],
                [1, 8, 2],
            ]
        )
        n_subQs = len(sorted_obj_values_with_split_subQs)
        subQs = list(range(n_subQs))

        (
            boundary_obj_values_all_subQs,
            boundary_configs_all_subQs,
        ), _ = dag_opt.get_boundary_all_subQs(
            subQs,
            sorted_obj_values_with_split_subQs,
            sorted_configs_with_split_subQs,
            sorted_indices_all_subQs,
        )

        expect_boundary_obj_values_all_subQs = [
            np.array([[22, 24], [25, 20]]),
            np.array([[23, 25], [23, 25]]),
            np.array([[27, 27], [27, 27]]),
            np.array([[23, 29], [28, 21]]),
        ]
        expect_boundary_configs_all_subQs = [
            np.array([[9, 4, 23, 24, 36, 1, 2], [9, 4, 26, 27, 34, 1, 2]]),
            np.array([[12, 16, 37, 25, 33, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 26, 27, 34, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
        ]
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    boundary_obj_values_all_subQs, expect_boundary_obj_values_all_subQs
                )
            ]
        )
        assert all(
            [
                np.all(result == expect)
                for result, expect in zip(
                    boundary_configs_all_subQs, expect_boundary_configs_all_subQs
                )
            ]
        )

    def test_construct_query_solutions(self, dag_opt: DAGOpt) -> None:
        boundary_obj_values_all_subQs = [
            np.array([[22, 24], [25, 20]]),
            np.array([[23, 25], [23, 25]]),
            np.array([[27, 27], [27, 27]]),
            np.array([[23, 29], [28, 21]]),
        ]
        boundary_configs_all_subQs = [
            np.array([[9, 4, 23, 24, 36, 1, 2], [9, 4, 26, 27, 34, 1, 2]]),
            np.array([[12, 16, 37, 25, 33, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 26, 27, 34, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
        ]
        n_subQs = 2
        n_theta_c = 2
        n_objs = 2
        len_theta = 7

        (
            boundary_query_objs,
            boundary_configs_all_subQs_arr,
        ), _ = dag_opt.compute_query_solutions(
            n_subQs,
            n_theta_c,
            n_objs,
            len_theta,
            boundary_obj_values_all_subQs,
            boundary_configs_all_subQs,
        )

        expect_boundary_query_objs = np.array([[49, 51], [52, 47], [46, 54], [51, 46]])
        expect_boundary_configs_all_subQs_arr = np.array(
            [
                [9, 4, 23, 24, 36, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                [9, 4, 26, 27, 34, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2, 12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2, 12, 16, 37, 25, 33, 1, 2],
            ]
        )

        assert np.all(boundary_query_objs == expect_boundary_query_objs)
        assert np.all(
            boundary_configs_all_subQs_arr == expect_boundary_configs_all_subQs_arr
        )

    def test_filter_dominated_boundary_based(self, dag_opt: DAGOpt) -> None:
        boundary_query_objs = np.array([[49, 51], [52, 47], [46, 54], [51, 46]])
        boundary_configs_all_subQs_arr = np.array(
            [
                [9, 4, 23, 24, 36, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                [9, 4, 26, 27, 34, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2, 12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2, 12, 16, 37, 25, 33, 1, 2],
            ]
        )

        (
            pareto_query_obj_values,
            pareto_query_configs,
        ), _ = dag_opt.filter_dominated_boundary_based(
            boundary_query_objs,
            boundary_configs_all_subQs_arr,
        )

        expect_pareto_query_obj_values = np.array([[46, 54], [49, 51], [51, 46]])
        expect_pareto_query_configs = np.array(
            [
                [12, 16, 37, 25, 33, 1, 2, 12, 16, 26, 27, 34, 1, 2],
                [9, 4, 23, 24, 36, 1, 2, 9, 4, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2, 12, 16, 37, 25, 33, 1, 2],
            ]
        )

        assert np.all(pareto_query_obj_values == expect_pareto_query_obj_values)
        assert np.all(pareto_query_configs == expect_pareto_query_configs)
