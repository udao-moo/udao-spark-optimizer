# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: test functions used in get_subQs_set
#
# Created at 18/04/2024

import numpy as np
import pandas as pd
import torch as th

from udao_spark.optimizer.moo_algos.hierarchical_moo_with_constraints import (
    Hierarchical_MOO_with_Constraints,
)

np.random.seed(0)
n_subQs = 3
graph_embeddings = th.tensor(np.random.randint(10, 15, (n_subQs, 2)), dtype=th.float32)
non_decision_tabular_features = pd.DataFrame(np.random.randint(10, 15, (n_subQs, 2)))
use_ag = True
cluster_representation = np.array([1, 7])
cluster_members = [np.array([1, 4, 5, 9]), np.array([0, 2, 3, 6, 7, 8])]

optimal_obj_values_all_subQs = np.array(
    [[25, 20], [23, 23], [23, 25], [22, 24], [21, 26], [27, 27], [28, 21], [25, 29]]
)
optimal_configurations_all_subQs = np.array(
    [
        [9, 4, 26, 27, 34, 1, 2],
        [9, 4, 37, 25, 33, 1, 2],
        [12, 16, 37, 25, 33, 1, 2],
        [9, 4, 26, 27, 34, 1, 2],
        [12, 16, 37, 25, 33, 1, 2],
        [9, 4, 26, 27, 34, 1, 2],
        [9, 4, 37, 25, 33, 1, 2],
        [12, 16, 26, 27, 34, 1, 2],
    ]
)
indices_all_subQs_theta_c_p = np.array(
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


class TestGetSubQsSet:
    def test_cluster_methods(self, hmooc: Hierarchical_MOO_with_Constraints) -> None:
        cluster_algo = "k-means"
        n_clusters = 2
        cluster_features = hmooc.theta_c_samples
        (cluster_labels, cluster_model), time_cost = hmooc.clustering_method(
            n_clusters, cluster_features, cluster_algo
        )

        expect_cluster_labels = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
        assert np.all(cluster_labels == expect_cluster_labels)

    def test_cluster_representation(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        cluster_labels = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
        # the returned values represent the indices of cluster labels
        (
            result_cluster_representation,
            result_cluster_members,
        ), _ = hmooc.cluster_representation(cluster_labels)

        assert np.all(result_cluster_representation == cluster_representation)
        for cm, expected_cm in zip(result_cluster_members, cluster_members):
            assert np.all(cm == expected_cm)

    def test_construct_model_input_for_opt_p(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        representative_c_indices = [1, 8]
        representative_c_features = hmooc.theta_c_samples[representative_c_indices]
        theta_p_samples = hmooc.theta_p_samples
        theta_s_samples = hmooc.theta_s_samples
        n_subQs = hmooc.n_subQs
        seed = hmooc.seed
        np.random.seed(seed)
        graph_embeddings = th.tensor(
            np.random.randint(10, 15, (n_subQs, 2)), dtype=th.float32
        )
        non_decision_tabular_features = pd.DataFrame(
            np.random.randint(10, 15, (n_subQs, 2))
        )
        use_ag = True

        (
            n_evals,
            mesh_theta,
            mesh_graph_embeddings,
            mesh_non_decision_tabular_features,
        ), time_cost = hmooc.construct_model_input_for_opt_p(
            representative_c_features,
            theta_p_samples,
            theta_s_samples,
            n_subQs,
            use_ag,
            graph_embeddings,
            non_decision_tabular_features,
        )

        expect_mesh_theta = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
            ]
        )
        expect_mesh_graph_embeddings = th.tensor(
            np.array(
                [
                    [14, 10],
                    [14, 10],
                    [14, 10],
                    [14, 10],
                    [13, 13],
                    [13, 13],
                    [13, 13],
                    [13, 13],
                    [13, 11],
                    [13, 11],
                    [13, 11],
                    [13, 11],
                ]
            ),
            dtype=th.float32,
        )
        expect_mesh_non_decision_tabular_features = pd.DataFrame(
            np.array(
                [
                    [13, 12],
                    [13, 12],
                    [13, 12],
                    [13, 12],
                    [14, 10],
                    [14, 10],
                    [14, 10],
                    [14, 10],
                    [10, 14],
                    [10, 14],
                    [10, 14],
                    [10, 14],
                ]
            )
        )
        expect_n_evals = expect_mesh_theta.shape[0]

        assert n_evals == expect_n_evals
        assert np.all(mesh_theta == expect_mesh_theta)
        assert np.all(
            mesh_graph_embeddings.numpy() == expect_mesh_graph_embeddings.numpy()
        )
        assert np.all(
            mesh_non_decision_tabular_features.to_numpy()
            == expect_mesh_non_decision_tabular_features.to_numpy()
        )

    def test_split_solutions(self, hmooc: Hierarchical_MOO_with_Constraints) -> None:
        representative_theta_c = [1, 8]
        n_representative_theta_c = len(representative_theta_c)
        n_theta_p_samples = hmooc.theta_p_samples.shape[0]
        n_subQs = hmooc.n_subQs
        mesh_theta = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
            ]
        )
        np.random.seed(hmooc.seed)
        y_hat = np.random.randint(20, 30, (mesh_theta.shape[0], 2))

        (split_y_hat_list, split_theta_list), _ = hmooc._split_solutions(
            y_hat, mesh_theta, n_subQs, n_representative_theta_c, n_theta_p_samples
        )

        expect_split_y_hat_list = [
            np.array([[25, 20], [23, 23]]),
            np.array([[27, 29], [23, 25]]),
            np.array([[22, 24], [27, 26]]),
            np.array([[28, 28], [21, 26]]),
            np.array([[27, 27], [28, 21]]),
            np.array([[25, 29], [28, 29]]),
        ]
        expect_split_theta_list = [
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
        ]
        for split_y_hat, expect_split_y_hat in zip(
            split_y_hat_list, expect_split_y_hat_list
        ):
            assert np.all(split_y_hat == expect_split_y_hat)

        for split_theta, expect_split_theta in zip(
            split_theta_list, expect_split_theta_list
        ):
            assert np.all(split_theta == expect_split_theta)

    def test_keep_optimal_theta_p_per_theta_c(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        representative_theta_c = [1, 8]
        ##          split_y_hat:                element  #
        ##################################################
        # subQ1 representative_theta_c1 #        array1
        # subQ1 representative_theta_c2 #        array2
        # subQ2 representative_theta_c1 #        array3
        # subQ2 representative_theta_c2 #        array4
        # subQ3 representative_theta_c1 #        array5
        # subQ3 representative_theta_c2 #        array6
        ##################################################
        split_y_hat_list = [
            np.array([[25, 20], [23, 23]]),
            np.array([[27, 29], [23, 25]]),
            np.array([[22, 24], [27, 26]]),
            np.array([[28, 28], [21, 26]]),
            np.array([[27, 27], [28, 21]]),
            np.array([[25, 29], [28, 29]]),
        ]
        split_theta_list = [
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
            np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
            np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
        ]

        (
            optimal_obj_values_subQ,
            optimal_configurations_subQ,
            indices_subQ_theta_c_p,
        ), _ = hmooc.keep_opt_theta_p_per_theta_c(
            2,
            representative_theta_c,
            split_y_hat_list[2],
            split_theta_list[2],
        )

        expect_optimal_obj_values_subQ = np.array([[22, 24]])
        expect_optimal_configurations_subQ = np.array([[9, 4, 26, 27, 34, 1, 2]])
        # it indicates the subQ2, theta_c1, and only 1 optimal theta_p_s
        expect_indices_subQ_theta_c_p = np.array([[1, 1, 0]])
        assert np.all(optimal_obj_values_subQ == expect_optimal_obj_values_subQ)
        assert np.all(optimal_configurations_subQ == expect_optimal_configurations_subQ)
        assert np.all(indices_subQ_theta_c_p == expect_indices_subQ_theta_c_p)

    def test_filter_dominated_theta_p_s(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        representative_theta_c = [1, 8]
        n_theta_p_samples = hmooc.theta_p_samples.shape[0]
        n_subQs = hmooc.n_subQs
        use_ag = True
        mesh_theta = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
            ]
        )
        np.random.seed(hmooc.seed)
        y_hat = np.random.randint(20, 30, (mesh_theta.shape[0], 2))

        ##          split_y_hat:                element  #
        ##################################################
        # subQ1 representative_theta_c1 #        array1
        # subQ1 representative_theta_c2 #        array2
        # subQ2 representative_theta_c1 #        array3
        # subQ2 representative_theta_c2 #        array4
        # subQ3 representative_theta_c1 #        array5
        # subQ3 representative_theta_c2 #        array6
        ##################################################
        # split_y_hat_list = [
        #     np.array([[25, 20], [23, 23]]),
        #     np.array([[27, 29], [23, 25]]),
        #     np.array([[22, 24], [27, 26]]),
        #     np.array([[28, 28], [21, 26]]),
        #     np.array([[27, 27], [28, 21]]),
        #     np.array([[25, 29], [28, 29]]),
        # ]
        # split_theta_list = [
        #     np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
        #     np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
        #     np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
        #     np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
        #     np.array([[9, 4, 26, 27, 34, 1, 2], [9, 4, 37, 25, 33, 1, 2]]),
        #     np.array([[12, 16, 26, 27, 34, 1, 2], [12, 16, 37, 25, 33, 1, 2]]),
        # ]

        # filter is addressed for each splitted array
        (
            optimal_obj_values_all_subQs,
            optimal_configurations_all_subQs,
            indices_all_subQs_theta_c_p,
        ), _ = hmooc.filter_dominated_theta_p_s(
            representative_theta_c,
            n_theta_p_samples,
            n_subQs,
            mesh_theta,
            y_hat,
            use_ag,
        )

        expect_optimal_obj_values_all_subQs = np.array(
            [
                [25, 20],
                [23, 23],
                [23, 25],
                [22, 24],
                [21, 26],
                [27, 27],
                [28, 21],
                [25, 29],
            ]
        )
        expect_optimal_configurations_all_subQs = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
            ]
        )
        expect_indices_all_subQs_theta_c_p = np.array(
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

        assert np.all(
            optimal_obj_values_all_subQs == expect_optimal_obj_values_all_subQs
        )
        assert np.all(
            optimal_configurations_all_subQs == expect_optimal_configurations_all_subQs
        )
        assert np.all(indices_all_subQs_theta_c_p == expect_indices_all_subQs_theta_c_p)

    def test_assign_optimal_theta_p_s(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        representative_theta_c = [1, 8]
        # for simplicity, just let each cluster has at most two members
        cluster_members = [np.array([1]), np.array([7, 8])]

        n_subQs = hmooc.n_subQs
        theta_c_samples = hmooc.theta_c_samples
        init_n_c_smaples = theta_c_samples.shape[0]
        seed = hmooc.seed
        np.random.seed(seed)
        graph_embeddings = th.tensor(
            np.random.randint(10, 15, (n_subQs, 2)), dtype=th.float32
        )
        non_decision_tabular_features = pd.DataFrame(
            np.random.randint(10, 15, (n_subQs, 2))
        )
        use_ag = True

        optimal_configurations_all_subQs = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
            ]
        )
        indices_all_subQs_theta_c_p = np.array(
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

        (
            result_theta,
            result_graph_embeddings,
            result_non_decision_tabular_features,
            result_indices_arr,
        ), _ = hmooc.assign_optimal_theta_p_s(
            representative_theta_c,
            init_n_c_smaples,
            cluster_members,
            theta_c_samples,
            optimal_configurations_all_subQs,
            indices_all_subQs_theta_c_p,
            n_subQs,
            use_ag,
            graph_embeddings,
            non_decision_tabular_features,
            mode="assign_initial",
        )

        expect_theta = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [10, 12, 37, 25, 33, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [10, 12, 37, 25, 33, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [10, 12, 26, 27, 34, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
            ]
        )
        expect_indices_arr = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 7, 0],
                [0, 8, 0],
                [1, 1, 0],
                [1, 7, 0],
                [1, 8, 0],
                [2, 1, 0],
                [2, 1, 1],
                [2, 7, 0],
                [2, 8, 0],
            ]
        )
        expect_graph_embeddings = th.tensor(
            [
                [14, 10],
                [14, 10],
                [14, 10],
                [14, 10],
                [13, 13],
                [13, 13],
                [13, 13],
                [13, 11],
                [13, 11],
                [13, 11],
                [13, 11],
            ]
        )
        expect_non_decision_tabular_features = np.array(
            [
                [13, 12],
                [13, 12],
                [13, 12],
                [13, 12],
                [14, 10],
                [14, 10],
                [14, 10],
                [10, 14],
                [10, 14],
                [10, 14],
                [10, 14],
            ]
        )

        assert np.all(result_theta == expect_theta)
        assert th.all(result_graph_embeddings == expect_graph_embeddings)
        assert np.all(
            result_non_decision_tabular_features.to_numpy()
            == expect_non_decision_tabular_features
        )
        assert np.all(result_indices_arr == expect_indices_arr)

    def test_union_optimal_theta_c(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        len_theta_c = hmooc.theta_c_samples.shape[1]
        union_theta_c_list, _ = hmooc.union_opt_theta_c(
            n_subQs,
            len_theta_c,
            optimal_obj_values_all_subQs,
            optimal_configurations_all_subQs,
            indices_all_subQs_theta_c_p,
        )

        expect_union_theta_c_list = [[9, 4], [12, 16]]
        assert np.all(
            np.array(union_theta_c_list) == np.array(expect_union_theta_c_list)
        )

    def test_assign_new_theta_c_to_clusters(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        cluster_algo = "k-means"
        n_clusters = 2
        cluster_features = hmooc.theta_c_samples
        (cluster_labels, cluster_model), time_cost = hmooc.clustering_method(
            n_clusters, cluster_features, cluster_algo
        )
        theta_c_samples = hmooc.theta_c_samples
        representative_theta_c_indices = [1, 8]
        new_theta_c_list = [[4, 5], [7, 8]]
        (
            new_cluster_members,
            new_representative_theta_c_indices,
        ), _ = hmooc.assign_new_theta_c_to_clusters(
            new_theta_c_list,
            theta_c_samples,
            representative_theta_c_indices,
            cluster_model,
        )

        expect_new_cluster_members = [[10, 11]]
        expect_new_representative_theta_c_indices = [1]
        assert np.all(
            np.array(new_cluster_members) == np.array(expect_new_cluster_members)
        )
        assert (
            new_representative_theta_c_indices
            == expect_new_representative_theta_c_indices
        )

    def test_crossover(self, hmooc: Hierarchical_MOO_with_Constraints) -> None:
        union_list = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ]
        cross_location = 3
        theta_c_samples = np.array(
            [
                [11, 23, 13, 45, 75],
                [62, 17, 82, 19, 12],
            ]
        )

        result_crossover, _ = hmooc.crossover(
            cross_location, union_list, theta_c_samples
        )

        expect_crossover = [[1, 2, 3, 9, 10], [6, 7, 8, 4, 5]]

        assert np.all(np.array(result_crossover) == np.array(expect_crossover))

    def test_union_initial_and_new_solution_sets(
        self, hmooc: Hierarchical_MOO_with_Constraints
    ) -> None:
        new_optimal_obj_values_all_subQs = np.array([[24, 21], [22, 23], [25, 20]])
        new_optimal_configurations_all_subQs = np.array(
            [
                [13, 14, 26, 27, 34, 1, 2],
                [13, 14, 37, 25, 33, 1, 2],
                [13, 14, 37, 25, 33, 1, 2],
            ]
        )
        new_indices_all_subQs_theta_c_p = np.array([[0, 9, 0], [1, 9, 0], [2, 9, 0]])
        model_inference_info_optimize_theta_p = np.array([100, 0.4])
        model_inference_info_assign_init_theta_c = np.array([200, 0.8])
        model_inference_info_assign_new_theta_c = np.array([150, 0.5])

        (
            obj_values_all_subQs,
            config_all_subQs,
            indices_all_subQs,
            final_model_inference_info,
        ), _ = hmooc.union_initial_and_new_solution_sets(
            optimal_obj_values_all_subQs,
            new_optimal_obj_values_all_subQs,
            optimal_configurations_all_subQs,
            new_optimal_configurations_all_subQs,
            indices_all_subQs_theta_c_p,
            new_indices_all_subQs_theta_c_p,
            model_inference_info_optimize_theta_p,
            model_inference_info_assign_init_theta_c,
            model_inference_info_assign_new_theta_c,
        )

        expect_obj_values_all_subQs = np.array(
            [
                [25, 20],
                [23, 23],
                [23, 25],
                [22, 24],
                [21, 26],
                [27, 27],
                [28, 21],
                [25, 29],
                [24, 21],
                [22, 23],
                [25, 20],
            ]
        )
        expect_config_all_subQs = np.array(
            [
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [12, 16, 37, 25, 33, 1, 2],
                [9, 4, 26, 27, 34, 1, 2],
                [9, 4, 37, 25, 33, 1, 2],
                [12, 16, 26, 27, 34, 1, 2],
                [13, 14, 26, 27, 34, 1, 2],
                [13, 14, 37, 25, 33, 1, 2],
                [13, 14, 37, 25, 33, 1, 2],
            ]
        )
        expect_indices_all_subQs = np.array(
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 8, 0],
                [1, 1, 0],
                [1, 8, 0],
                [2, 1, 0],
                [2, 1, 1],
                [2, 8, 0],
                [0, 9, 0],
                [1, 9, 0],
                [2, 9, 0],
            ]
        )
        expect_final_model_inference_info = np.array(
            [[100.0, 0.4], [200.0, 0.8], [150.0, 0.5]]
        )

        assert np.all(obj_values_all_subQs == expect_obj_values_all_subQs)
        assert np.all(config_all_subQs == expect_config_all_subQs)
        assert np.all(indices_all_subQs == expect_indices_all_subQs)
        assert np.all(final_model_inference_info == expect_final_model_inference_info)
