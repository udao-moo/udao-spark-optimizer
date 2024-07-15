# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Algorithm of Hierarchical MOO with Constraints
#
# Created at 18/04/2024

import random
import signal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from sklearn.cluster import KMeans
from udao.optimization.utils.moo_utils import is_pareto_efficient

from udao_spark.optimizer.moo_algos.dag_optimization import DAGOpt
from udao_spark.optimizer.utils import timeis


class Hierarchical_MOO_with_Constraints:
    def __init__(
        self,
        n_subQs: int,
        theta_c_samples: Union[np.ndarray, th.Tensor],
        theta_p_samples: Union[np.ndarray, th.Tensor],
        theta_s_samples: Union[np.ndarray, th.Tensor],
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialization of the Hierarchical MOO with Constraints (HMOOC) approach
        Including the number of subQs, the initial samples of theta_c, theta_p,
        theta_s and the random seed used
        :param n_subQs: the number of subQs
        :param theta_c_samples: the theta_c samples
        :param theta_p_samples: the theta_p samples
        :param theta_s_samples:the theta_s samples
        :param seed: random seed used in the approach
        """

        # for the HMOOC algorithm
        self.n_subQs = n_subQs
        self.theta_c_samples = theta_c_samples
        self.theta_p_samples = theta_p_samples
        self.theta_s_samples = theta_s_samples
        self.len_theta_c = theta_c_samples.shape[1]
        # the length of one copy of theta
        self.len_theta = (
            theta_c_samples.shape[1]
            + theta_p_samples.shape[1]
            + theta_s_samples.shape[1]
        )
        self.n_objs = 2
        self.seed = seed

        all_function_names = [
            method
            for method in Hierarchical_MOO_with_Constraints.__dict__
            if callable(getattr(Hierarchical_MOO_with_Constraints, method))
            and not method.startswith("__")
        ]
        self.global_time_cost: dict
        self.global_time_cost = {k: [] for k in all_function_names}

    def model_setup(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: Union[pd.DataFrame, th.Tensor],
        obj_model: Callable[[Any, Any, Any, Any], Any],
        use_ag: bool,
        ag_model: Dict[str, str],
    ) -> None:
        """
        initialize the setup for predictive models
        :param graph_embeddings: graph embeddings for the model, shape (n_subQs, 32)
        :param non_decision_tabular_features: non-decision tabular features: shape
                (n_subQs, 28)
        :param obj_model: the function calling predictive models to return latency
                and cost values, with input of graph embeddings, non-decision
                tabular features, theta and model name
                (e.g. dict() = {"ana_latency_s": "WeightedEnsemble_L2_FULL",
                "io_mb":"CatBoost"})
        :param use_ag: flag to indicate whether using AutoGluon in models
        """

        # for the predictive models
        self.graph_embeddings = graph_embeddings
        self.non_decision_tabular_features = non_decision_tabular_features
        self.obj_model = obj_model
        self.ag_model = ag_model
        self.use_ag = use_ag
        self.device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

    def hmooc_setup(
        self,
        theta_sample_mode: str,
        theta_sample_funcs: Callable[..., Any],
        n_clusters: int,
        cross_location: int,
        dag_opt_algo: str,
        cluster_algo: str = "k-means",
        runmode: str = "multiprocessing",
        time_limit: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        initialize input related to the Hierarchical Mulit-Objective Optimization
        with Constraints (HMOOC) approach.
        :param theta_sample_mode: the mode to indicate how to get the initial theta
                samples, e.g. "grid-search", "random"
        :param theta_sample_funcs: the function passed to generate the "random"
                initial samples for theta.
                it takes inputs
                1) the #samples (int),
                2) type (str, e.g. "c" indicates for theta_c, similarly for "p"
                    and "s"),
                3) seed (int),
                4) whether to normalize (bool),
                5) sampling mode: by defaul, random sampling, can also be set to "lhs"
                and it returns an np.ndarray with shape (#samples, len_theta_c/p/s).
                If using AutoGulon,  normalize = False, otherwise (i.e. using MLP),
                normalize = True
        :param n_clusters: the number of clusters used to cluster theta_c samples
        :param cross_location: the location of crossover, e.g. the following is the
                case of cross_location=3
                ###############|##########################
                # k1 # k2 # k3 |# k4 # k5 # k6 # k7 # k8 #
                # k1'# k2'# k3'|# k4'# k5'# k6'# k7'# k8'#
                ###############|##########################
        :param dag_opt_algo: the DAG optimization method, including:
                - GD: General Divide-and-conquer;
                - WS&int: Weighted Sum (WS)-based method, int is an integer showing
                    the number of weights used (e.g. 11);
                - B: Boundary-based method
        :param cluster_algo: the clustering method used to cluster theta_c samples,
                e.g. "k-means"
        :param runmode: to indicate whether using multiprocessing
        :param time_limit: the time limit with unit of seconds. e.g. 60, which means
                the function stops running at 60s, "-1" means no time limit.
        :param verbose: the flag to indicate whether to print more information
                during running
        """
        self.theta_sample_mode = theta_sample_mode
        self.theta_sample_funcs = theta_sample_funcs
        self.cluster_algo = cluster_algo
        self.n_clusters = n_clusters
        self.cross_location = cross_location
        self.dag_opt_algo = dag_opt_algo

        # related to algorithm settings
        self.runmode = runmode
        self.time_limit = time_limit
        self.verbose = verbose

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """fixme: to double-check whether runnable
        The entry of the Hierarchical Mulit-Objective Optimization with Constraints
         (HMOOC) approach.
        It returns the query-level Pareto optimal objective values, the
        corresponding configurations (i.e. theta) of all subQs, and the model
        inference information during running (e.g. the number of evaluations, the
        time cost)
        :return: query_obj_values: shape (#solutions, n_objs), here n_objs = 2, with
                        latency and cost
                query_configurations: configurations of all subQs, shape
                        (#solutions, len_one_theta * n_subQs)
                model_inference_info: shape (3, 2), where 3 means HMOOC calls the
                        predictive model 3 times; 2 shows two columes
                        "n_evaluations" and "inference_time" respectively,
                        - e.g. when 100 * len_one_theta is fed in the model,
                        n_evaluations = 100
        """
        if self.time_limit > 0:
            # function used for time limit
            def signal_handler(signum: Any, frame: Any) -> None:
                raise Exception("Timed out!")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.time_limit)

            try:
                (
                    query_obj_values,
                    query_configurations,
                    model_inference_info,
                    dag_opt_global_time_cost,
                ) = self._hmooc()
                signal.alarm(
                    0
                )  ##cancel the timer if the function returned before timeout

            except Exception:
                query_obj_values = -1 * np.ones((1, self.n_objs))
                query_configurations = np.array([[-1], [-1]])
                model_inference_info = np.array([-1])
                dag_opt_global_time_cost = {}
                print("Timed out for query")
        else:
            (
                query_obj_values,
                query_configurations,
                model_inference_info,
                dag_opt_global_time_cost,
            ) = self._hmooc()

        total_time_profile = {
            "hmooc": self.global_time_cost,
            "dag_opt": dag_opt_global_time_cost,
        }
        return (
            query_obj_values,
            query_configurations,
            model_inference_info,
            total_time_profile,
        )

    def _hmooc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """fixme: to double-check whether runnable
        The entry of the Hierarchical Mulit-Objective Optimization with Constraints
         (HMOOC) approach.
        It returns the query-level Pareto optimal objective values, the
        corresponding configurations (i.e. theta) of all subQs, and the model
        inference information during running (e.g. the number of evaluations, the
        time cost)
        :return: query_obj_values: shape (#solutions, n_objs), here n_objs = 2,
                        with latency and cost
                query_configurations: shape (#solutions, len_one_theta * n_subQs)
                model_inference_info: shape (3, 2), where 3 means HMOOC calls the
                        predictive model 3 times; 2 shows two columes
                        "n_evaluations" and "inference_time" respectively,
                        - e.g. when 100 * len_one_theta is fed in the model,
                        n_evaluations = 100
        """

        (
            subQ_obj_values,
            subQ_configurations,
            indices_all_subQs,
            model_inference_info,
        ), tc_get_subQs_sets = self._get_subQ_sets()
        self.global_time_cost[self._get_subQ_sets.__name__].append(tc_get_subQs_sets)

        (
            query_obj_values,
            query_configurations,
            dag_opt_global_time_cost,
        ), tc_dag_opt = self._dag_opt(
            subQ_obj_values,
            subQ_configurations,
            indices_all_subQs,
        )
        self.global_time_cost[self._dag_opt.__name__].append(tc_dag_opt)

        return (
            query_obj_values,
            query_configurations,
            model_inference_info,
            dag_opt_global_time_cost,
        )

    @timeis
    def _get_subQ_sets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """fixme: to double-check whether runnable
        It searches for the optimal subQ-level solutions under the constraint that
        theta_c is identical among all subQs.
        :return: obj_values_all_subQs: the subQ-level Pareto optimal solutions,
                    which are Pareto optimal under each theta_c and subQ, shape
                    (#total_solutions, n_objs), here n_objs=2 with latency and cost.
                config_all_subQs: the corresponding configurations, shape
                    (#total_solutions, len_one_theta), here
                    len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
                indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
                    for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                    - e.g. np.array([[0, 1, 2]]) indicates the second solution of
                    subQ0 under theta_c_1 (indexing from 0).
                final_model_inference_info: shape (3, 2), where 3 means HMOOC calls
                    the predictive model 3 times; 2 shows two columes
                    "n_evaluations" and "inference_time" respectively,
                    - e.g. when 100 * len_one_theta is fed in the model,
                    n_evaluations = 100
        """

        # 1. get samples of theta_c, theta_p and theta_s
        theta_c_samples = self.theta_c_samples
        theta_p_samples = self.theta_p_samples
        theta_s_samples = self.theta_s_samples

        # 2. cluster_based optimal theta_p and theta_p_s estimation
        # 2.1 cluster theta_c
        (clusters_labels, cluster_model), tc_cluster = self.clustering_method(
            self.n_clusters, theta_c_samples, cluster_algo=self.cluster_algo
        )
        self.global_time_cost[self.clustering_method.__name__].append(tc_cluster)

        (
            representative_theta_c,
            cluster_members,
        ), tc_cluster_representation = self.cluster_representation(clusters_labels)
        self.global_time_cost[self.cluster_representation.__name__].append(
            tc_cluster_representation
        )

        # 2.2. optimize theta_p and theta_s of the representative theta_c samples
        # from clusters for all subQs
        (
            obj_values_all_subQs_rep_theta_c,
            configs_all_subQs_rep_theta_c,
            indices_all_subQs_rep_theta_c,
            model_inference_info_optimize_theta_p,
        ), tc_optimize_theta_p_s = self._optimize_theta_p_and_s(
            representative_theta_c,
            theta_c_samples[representative_theta_c],
            theta_p_samples,
            theta_s_samples,
            self.n_subQs,
            self.use_ag,
        )
        self.global_time_cost[self._optimize_theta_p_and_s.__name__].append(
            tc_optimize_theta_p_s
        )

        # 2.3. estimate optimal theta_p and theta_s for all theta_c samples
        # i.e. assign the optimal theta_p and theta_s of the representative theta_c
        # to all memebers within the same cluster
        (
            obj_values_all_subQs_init_theta_c,
            config_all_subQs_init_theta_c,
            indices_all_subQs_init_theta_c,
            model_inference_info_assign_init_theta_c,
        ), tc_estimate_init_opt_theta_p_s = self.estimate_opt_theta_p_s(
            representative_theta_c=representative_theta_c,
            init_n_c_samples=theta_c_samples.shape[0],
            cluster_members=cluster_members,
            theta_c_samples=theta_c_samples,
            configurations_all_subQs=configs_all_subQs_rep_theta_c,
            indices_all_subQs=indices_all_subQs_rep_theta_c,
            n_subQs=self.n_subQs,
            use_ag=self.use_ag,
            graph_embeddings=self.graph_embeddings,
            non_decision_tabular_features=self.non_decision_tabular_features,
            mode="assign_initial",
        )
        self.global_time_cost[self.estimate_opt_theta_p_s.__name__].append(
            tc_estimate_init_opt_theta_p_s
        )

        # 3. theoretical result 2: get local optimal theta_c of each subQ and union
        # them all
        union_opt_theta_c_list, tc_union_optimal_theta_c = self.union_opt_theta_c(
            self.n_subQs,
            self.len_theta_c,
            obj_values_all_subQs_init_theta_c,
            config_all_subQs_init_theta_c,
            indices_all_subQs_init_theta_c,
        )
        self.global_time_cost[self.union_opt_theta_c.__name__].append(
            tc_union_optimal_theta_c
        )

        # 4. extend new theta_c samples
        new_theta_c_list, tc_extend_c = self.extend_c(
            self.cross_location,
            union_opt_theta_c_list.copy(),
            self.use_ag,
            self.theta_sample_mode,
            self.theta_c_samples,
        )
        self.global_time_cost[self.extend_c.__name__].append(tc_extend_c)

        if self.verbose:
            print(
                f"the number of new generated theta_c candidates "
                f"is {len(new_theta_c_list)}"
            )
            print()

        # 5. cluster-based theta_p_s estimation
        (
            new_cluster_members,
            new_representative_theta_c,
        ), tc_assign_new_theta_c_to_clusters = self.assign_new_theta_c_to_clusters(
            new_theta_c_list,
            theta_c_samples,
            representative_theta_c,
            cluster_model,
        )
        self.global_time_cost[self.assign_new_theta_c_to_clusters.__name__].append(
            tc_assign_new_theta_c_to_clusters
        )

        (
            obj_values_all_subQs_new_theta_c,
            config_all_subQs_new_theta_c,
            indices_all_subQs_new_theta_c,
            model_inference_info_assign_new_theta_c,
        ), tc_estimate_new_opt_theta_p_s = self.estimate_opt_theta_p_s(
            representative_theta_c=new_representative_theta_c,
            init_n_c_samples=theta_c_samples.shape[0],
            cluster_members=new_cluster_members,
            theta_c_samples=np.array(new_theta_c_list),
            configurations_all_subQs=configs_all_subQs_rep_theta_c,
            indices_all_subQs=indices_all_subQs_rep_theta_c,
            n_subQs=self.n_subQs,
            use_ag=self.use_ag,
            graph_embeddings=self.graph_embeddings,
            non_decision_tabular_features=self.non_decision_tabular_features,
            mode="assign_new",
        )
        self.global_time_cost[self.estimate_opt_theta_p_s.__name__].append(
            tc_estimate_new_opt_theta_p_s
        )

        # 6. union all solutions
        (
            obj_values_all_subQs,
            config_all_subQs,
            indices_all_subQs,
            final_model_inference_info,
        ), tc_union_all_solutions = self.union_initial_and_new_solution_sets(
            obj_values_all_subQs_init_theta_c,
            obj_values_all_subQs_new_theta_c,
            config_all_subQs_init_theta_c,
            config_all_subQs_new_theta_c,
            indices_all_subQs_init_theta_c,
            indices_all_subQs_new_theta_c,
            model_inference_info_optimize_theta_p,
            model_inference_info_assign_init_theta_c,
            model_inference_info_assign_new_theta_c,
        )
        self.global_time_cost[self.union_initial_and_new_solution_sets.__name__].append(
            tc_union_all_solutions
        )

        return (
            obj_values_all_subQs,
            config_all_subQs,
            indices_all_subQs,
            final_model_inference_info,
        )

    @timeis
    def _dag_opt(
        self,
        obj_values_all_subQs: np.ndarray,
        config_all_subQs: Union[th.Tensor, np.ndarray],
        indices_all_subQs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Intialize the DAGOpt class and call the solve function to return the
        query-level Pareto optimal solutions recovered from the subQ-level Pareto
        optimal solutions with identical theta_c constraints.
        :param obj_values_all_subQs: the subQ-level Pareto optimal solutions, which
                are Pareto optimal under each theta_c and subQ, shape
                (#total_solutions, n_objs), here n_objs=2 with latency and cost.
        :param config_all_subQs: the corresponding configurations, shape
                (#total_solutions, len_one_theta), here
                len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
        :param indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
                for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                - e.g. np.array([[0, 1, 2]]) indicates the second solution of subQ0
                 under theta_c_1 (indexing from 0).
        :return: query_obj_values: the query-level Pareto optimal solutions, shape
                        (#solutions, n_objs), here n_objs=2
                query_configurations: configurations of all subQs, shape
                        (#solutions, len_one_theta * n_subQs)
        """
        dag_opt = DAGOpt(
            obj_values_all_subQs,
            config_all_subQs,
            indices_all_subQs,
            self.dag_opt_algo,
            self.runmode,
        )

        (
            query_obj_values,
            query_configurations,
            dag_opt_global_time_cost,
        ) = dag_opt.solve()
        return query_obj_values, query_configurations, dag_opt_global_time_cost

    ############## Sub-functions used in the function _get_subQ_sets #############
    @timeis
    def clustering_method(
        self,
        n_cluster: int,
        cluster_features: Union[th.Tensor, np.ndarray],
        cluster_algo: str,
    ) -> Tuple[Any, object]:
        """
        cluster initial theta_c samples
        e.g. 6 theta_c samples are clustered into 3 groups,
        np.array([0, 1, 1, 2, 0, 0])
        :param n_cluster: the number of clusters
        :param cluster_features: the samples to be clustered, with shape
                (n_samples, n_dim)
        :param cluster_algo: the clustering method used, currently it is "k-means"
        :return: the cluster labels for all samples
        """
        if cluster_algo == "k-means":
            k_means = KMeans(
                n_clusters=n_cluster, random_state=0, n_init="auto", algorithm="elkan"
            ).fit(cluster_features)
            clustering_labels = k_means.labels_
            return clustering_labels, k_means
        else:
            raise Exception(f"Clustering method {cluster_algo} is not supported!")

    @timeis
    def cluster_representation(
        self, cluster_labels: np.ndarray
    ) -> Tuple[List[Any], List[List[Any]]]:
        """# fixme: double-check the refactoring
        the randomly select one representative cluster label for each cluster,
        where each value represents the index of the cluster label
        e.g. 6 labels are grouped into 3 clusters, np.array([0, 1, 1, 2, 0, 0])
        :param cluster_labels: cluster labels, where it includes n elements, and
                each represents a cluster label
        :return: the cluster representations (e.g. 3 clusters, [4, 1, 3])
                and the cluster members (e.g. [np.array([0, 4, 5)],
                np.array([1, 2]), np.array([3])])
        """

        # use the indices of cluster labels to indicate the indices of theta_c
        n_clusters = np.unique(np.array(cluster_labels))
        n_clusters_init = cluster_labels

        new_clusters = []
        new_cluster_members = []

        label_rep_theta_c_mapping = dict()

        for index, item in enumerate(n_clusters):
            indexes = [j for j, x in enumerate(n_clusters_init) if x == item]
            inst_cluster_pick_representative = "random"
            if inst_cluster_pick_representative == "random":
                # randomly choose one instance to represent the current cluster
                random.seed(len(cluster_labels))
                random_choice = random.choice(indexes)
                new_inst_index = [random_choice]
                new_cluster_members.append(indexes)

                new_clusters.extend(new_inst_index)
                label_rep_theta_c_mapping[item] = new_inst_index[0]
            else:
                raise NotImplementedError(inst_cluster_pick_representative)

        representative_theta_c = list(label_rep_theta_c_mapping.values())

        return representative_theta_c, new_cluster_members

    @timeis
    def construct_model_input_for_opt_p(
        self,
        represent_theta_c_features: Union[np.ndarray, th.Tensor],
        theta_p_samples: Union[np.ndarray, th.Tensor],
        theta_s_samples: Union[np.ndarray, th.Tensor],
        n_subQs: int,
        use_ag: bool,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: Union[th.Tensor, pd.DataFrame],
    ) -> Tuple[
        int, Union[np.ndarray, th.Tensor], th.Tensor, Union[th.Tensor, pd.DataFrame]
    ]:
        """# fixme: double-check the refactoring
        construct the model input from the representative theta_c features, initial
         theta_p and theta_s samples, where the model input includes theta_c,
         theta_p, theta_s, graph_embeddings and non_decision_tabular_features.
        Graph embeddings and non_decision_tabular_features are with shape
        (n_subQs, x), which indicates that they are the same in the same subQ and
        different in different subQs;
        e.g. to get ONE predictive value of the subQ-level predictive model, it
        should include:
        one theta_c (1, 8), one theta_p (1, 9), one theta_s (1, 2),
        graph_embeddings (1, 32) and non_decision_tabular_feature (1, 28)
        The returned theta follows the orders of theta_c, theta_p and theta_s
        repsectively.
            e.g. given 2 theta_c_samples, 3 theta_p samples and 1 theta_s samples,
            the returned theta is:
            ################################
            # theta_c1, theta_p1, theta_s1 #
            # theta_c1, theta_p2, theta_s1 #
            # theta_c1, theta_p3, theta_s1 #
            # theta_c2, theta_p1, theta_s1 #
            # theta_c2, theta_p2, theta_s1 #
            # theta_c2, theta_p3, theta_s1 #
            ################################
        :param represent_theta_c_features: the features of representative theta_c
        :param theta_p_samples: theta_p samples
        :param theta_s_samples: theta_s samples
        :param n_subQs: the number of subQs
        :param use_ag: model flag, to indicate whether to use AutoGluon in the
                predictive model
        :param graph_embeddings: input of the model, with shape (n_subQs, 32)
        :param non_decision_tabular_features: input of the model, with shape
                (n_subQs, 28)
        :return: the number of evaluations fed in the model (i.e. how many
                 predictions it generates), the full theta, graph_embeddings
                 and non_decision_tabular_features for all subQs
        """
        # the number of evaluations fed in the model
        n_evals = (
            represent_theta_c_features.shape[0] * theta_p_samples.shape[0] * n_subQs
        )

        mesh_c: Union[np.ndarray, th.Tensor]
        mesh_p: Union[np.ndarray, th.Tensor]
        mesh_s: Union[np.ndarray, th.Tensor]

        if isinstance(represent_theta_c_features, np.ndarray):
            assert isinstance(theta_p_samples, np.ndarray)
            assert isinstance(theta_s_samples, np.ndarray)
            mesh_c_per_stage = np.repeat(
                represent_theta_c_features, theta_p_samples.shape[0], axis=0
            )
            mesh_c = np.tile(mesh_c_per_stage, (n_subQs, 1))

            mesh_p_per_stage = np.tile(
                theta_p_samples, (represent_theta_c_features.shape[0], 1)
            )
            mesh_p = np.tile(mesh_p_per_stage, (n_subQs, 1))

            mesh_s = np.repeat(theta_s_samples, n_evals, axis=0)
            assert (
                mesh_c.shape[0] == mesh_p.shape[0]
                and mesh_s.shape[0] == mesh_p.shape[0]
            )
            mesh_theta = np.concatenate([mesh_c, mesh_p, mesh_s], axis=1)
        else:
            assert isinstance(theta_p_samples, th.Tensor)
            assert isinstance(theta_s_samples, th.Tensor)
            assert isinstance(represent_theta_c_features, th.Tensor)
            mesh_c = represent_theta_c_features.repeat_interleave(
                theta_p_samples.shape[0], dim=0
            ).repeat(n_subQs, 1)
            mesh_p = theta_p_samples.repeat(
                represent_theta_c_features.shape[0], 1
            ).repeat(n_subQs, 1)
            mesh_s = theta_s_samples.repeat(n_evals, 1)
            assert (
                mesh_c.shape[0] == mesh_p.shape[0]
                and mesh_s.shape[0] == mesh_p.shape[0]
            )
            mesh_theta = th.cat([mesh_c, mesh_p, mesh_s], dim=1)

        mesh_graph_embeddings = graph_embeddings.repeat_interleave(
            theta_p_samples.shape[0] * represent_theta_c_features.shape[0], dim=0
        )

        mesh_non_decision_tabular_features: Union[th.Tensor, pd.DataFrame]
        if use_ag:
            assert isinstance(non_decision_tabular_features, pd.DataFrame)

            mesh_non_decision_tabular_features = non_decision_tabular_features.loc[
                np.repeat(
                    non_decision_tabular_features.index,
                    theta_p_samples.shape[0] * represent_theta_c_features.shape[0],
                )
            ].reset_index(drop=True)
        else:
            assert isinstance(non_decision_tabular_features, th.Tensor)
            mesh_non_decision_tabular_features = (
                non_decision_tabular_features.repeat_interleave(
                    theta_p_samples.shape[0] * represent_theta_c_features.shape[0],
                    dim=0,
                )
            )
        assert mesh_theta.shape[0] == n_evals
        assert mesh_theta.shape[0] == mesh_graph_embeddings.shape[0]
        assert mesh_theta.shape[0] == mesh_non_decision_tabular_features.shape[0]

        return (
            n_evals,
            mesh_theta,
            mesh_graph_embeddings,
            mesh_non_decision_tabular_features,
        )

    @timeis
    def get_obj_values(
        self,
        n_evals: int,
        mesh_theta: th.Tensor,
        mesh_graph_embeddings: th.Tensor,
        mesh_non_decision_tabular_features: Union[th.Tensor, pd.DataFrame],
    ) -> np.ndarray:
        """
        predict objective values for all subQs
        :param n_evals: the number of rows of the model input
            i.e. it equals to the number of predictions from the model
        :param mesh_graph_embeddings: the graph embeddings for all subQs
        :param mesh_non_decision_tabular_features: the
                non_decision_tabular_features for all subQs
        :param mesh_theta: the theta for all subQs
        :return: the model predictions, array with shape (n_evals, n_objs)
        """
        if self.verbose:
            print(f"n_evals is {n_evals}")
        if self.use_ag:
            y_hat = self.obj_model(
                mesh_graph_embeddings.cpu().numpy(),
                mesh_non_decision_tabular_features,
                mesh_theta,
                self.ag_model,
            )
        else:
            y_hat = self.obj_model(
                mesh_graph_embeddings,
                mesh_non_decision_tabular_features,
                mesh_theta,
                None,
            )
        return y_hat

    @timeis
    def filter_dominated_theta_p_s(
        self,
        representative_theta_c: list,
        n_theta_p_samples: int,
        n_subQs: int,
        mesh_theta: np.ndarray,
        y_hat: np.ndarray,
        use_ag: bool,
    ) -> Tuple[np.ndarray, Union[np.ndarray, th.Tensor], np.ndarray]:
        """fixme: to double-check
        for each subQ, filter dominated solutions under each theta_c, as theta_c
        should be the same among all subQs
        :param representative_theta_c: the theta_c indices of representative theta_c
                of theta_c clusters
        :param n_theta_p_samples: the number of initial theta_p samples
        :param n_subQs: the number of subQs
        :param mesh_theta: the input theta for all subQs,
               with shape (n_subQs * n_representative_theta_c * n_theta_p_samples,
               (len_theta_c+len_theta_p+len_theta_s)
        :param y_hat: the predictions with input mesh theta and graph embeddings
                and non-decision tabular features
        :param use_ag: model flag, to indicate whether to use AutoGluon in the
                predictive model
        :return: optimal solutions (subQ-level objective values and corresponding
                configurations) under each theta_c for all subQs

        """

        # each element in the list is an array with shape (n_theta_p_samples, n_objs)
        # each element represents the objective values with fixed subQ and theta_c
        # and different theta_p samples.
        # the length of the list is n_subQs * n_representative_theta_c
        # e.g. with 2 subQs and 3 representative theta_c, it splits into 6 elements,
        # where each is an array
        ##          split_y_hat:                element  #       optimal   #
        ####################################################################
        # subQ0 representative_theta_c0 #        array0  #      array0*    #
        # subQ0 representative_theta_c1 #        array1  #      array1*    #
        # subQ0 representative_theta_c2 #        array2  #      array2*    #
        # subQ1 representative_theta_c0 #        array3  #      array3*    #
        # subQ1 representative_theta_c1 #        array4  #      array4*    #
        # subQ1 representative_theta_c2 #        array5  #      array5*    #
        ####################################################################
        ##       optimal indices_all_subQs_theta_c_p     ##
        # subQ_id # theta_c_id  # opt_theta_p_id          #
        # 0       # 0           # range(array0*.shape[0]) #
        # 0       # 1           # range(array1*.shape[0]) #
        # 0       # 2           # range(array2*.shape[0]) #
        # 1       # 0           # range(array3*.shape[0]) #
        # 1       # 1           # range(array4*.shape[0]) #
        # 1       # 2           # range(array5*.shape[0]) #
        ###################################################

        (split_y_hat, split_theta), tc_split_theta = self._split_solutions(
            y_hat,
            mesh_theta,
            n_subQs,
            len(representative_theta_c),
            n_theta_p_samples,
        )
        self.global_time_cost[self._split_solutions.__name__].append(tc_split_theta)

        optimal_obj_values_per_subQ_theta_c = []
        optimal_indices_subQ_theta_c_p = []
        optimal_configurations_per_subQ_theta_c: List[Any] = []
        # fixme: to improve efficiency
        for i, (sub_obj_values, sub_config) in enumerate(zip(split_y_hat, split_theta)):
            # i: is the index of the list
            # each sub_obj_values/sub_config is under the same subQ and same theta_c

            (
                opt_obj_values,
                opt_configurations,
                opt_indices,
            ), tc_keep_opt_theta_p = self.keep_opt_theta_p_per_theta_c(
                i, representative_theta_c, sub_obj_values, sub_config
            )
            self.global_time_cost[self.keep_opt_theta_p_per_theta_c.__name__].append(
                tc_keep_opt_theta_p
            )

            optimal_obj_values_per_subQ_theta_c.append(opt_obj_values)
            optimal_configurations_per_subQ_theta_c.append(opt_configurations)
            # each opt_indices is with shape(n_opt_theta_p, 3)
            # the columes are: subQ_id, theta_c_id, optimal_theta_p_id (from 0 to
            # n_optimal_theta_p)
            optimal_indices_subQ_theta_c_p.append(opt_indices)

        if use_ag:
            assert isinstance(optimal_configurations_per_subQ_theta_c[0], np.ndarray)
            optimal_configurations_all_subQs = np.concatenate(
                optimal_configurations_per_subQ_theta_c
            )
        else:
            assert isinstance(optimal_configurations_per_subQ_theta_c[0], th.Tensor)
            optimal_configurations_all_subQs = th.concatenate(
                optimal_configurations_per_subQ_theta_c
            )

        optimal_obj_values_all_subQs = np.concatenate(
            optimal_obj_values_per_subQ_theta_c
        )
        indices_all_subQs_theta_c_p = np.concatenate(optimal_indices_subQ_theta_c_p)

        return (
            optimal_obj_values_all_subQs,
            optimal_configurations_all_subQs,
            indices_all_subQs_theta_c_p,
        )

    @timeis
    def _split_solutions(
        self,
        y_hat: np.ndarray,
        mesh_theta: np.ndarray,
        n_subQs: int,
        n_representative_theta_c: int,
        n_theta_p_samples: int,
    ) -> Tuple[List[np.ndarray], Sequence[Union[np.ndarray, th.Tensor]]]:
        """fixme: to double-check
        split the model predictions to generate a list, where each element has the
        same subQ_id and theta_c_id
        :param y_hat: model predictions
        :param mesh_theta: the input theta for all subQs,
               with shape (n_subQs * n_representative_theta_c * n_theta_p_samples,
               (len_theta_c+len_theta_p+len_theta_s)
        :param n_subQs: the number of subQs
        :param n_representative_theta_c: the number of representative theta_c
        :param n_theta_p_samples: the number of the initial theta_p samples
        :return: split_y_hat: list, where each item is an array with shape
                        (n_theta_p_samples, n_objs)
                split_theta: list, where each item is an array with shape
                        (n_theta_p_samples, len_one_theta)
        """
        split_y_hat: List[np.ndarray]
        split_theta: Sequence[Union[np.ndarray, th.Tensor]]

        if isinstance(mesh_theta, np.ndarray):
            # case for AutoGluon
            split_times = n_subQs * n_representative_theta_c
            split_y_hat = np.split(y_hat, split_times)
            split_theta = np.split(mesh_theta, split_times)
        else:
            # case for MLP
            assert isinstance(mesh_theta, th.Tensor)
            split_y_hat = np.split(y_hat, n_subQs * n_representative_theta_c)
            split_theta = th.split(mesh_theta, n_theta_p_samples)

        # to double-check each split has the same theta_c and different theta_p
        assert all(
            [
                np.unique(x[:, : self.len_theta_c], axis=0).shape[0] == 1
                for x in split_theta
            ]
        )

        return split_y_hat, split_theta

    @timeis
    def keep_opt_theta_p_per_theta_c(
        self,
        i: int,
        representative_theta_c: list,
        sub_obj_values: np.ndarray,
        sub_config: Union[th.Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, Union[th.Tensor, np.ndarray], np.ndarray]:
        """# fixme: to double-check
        to filter dominated solution under a fixed theta_c for a specified subQ
        :param i: i is the index of the arrays.
            e.g. in the following example, i \in range([0, 6])
        :param representative_theta_c: indices of representative theta_c in theta_c
                clusters
        :param sub_obj_values: the objective values under a fixed theta_c for a
                specified subQ
        :param sub_config: the configurations under a fixed theta_c for a specified
                subQ
        :return: po_objs: array, the Pareto optimal solutions
                po_configs: array, the corresponding configurations
                indices_arr: array, indices of subQ_id, theta_c_id and optimal
                        theta_p_id
                    e.g. np.array([[0, 1, 0], [0, 1, 1], [0, 1, 2]])
                    it represents that for subQ0 with theta_c1, there are 3
                    optimal solutions
        """
        # e.g. with 2 subQs and 3 representative theta_c, it splits into 6 elements,
        # where each is an array
        ##          split_y_hat:                element  #
        ##################################################
        # subQ0 representative_theta_c0 #        array0
        # subQ0 representative_theta_c1 #        array1
        # subQ0 representative_theta_c2 #        array2
        # subQ1 representative_theta_c0 #        array3
        # subQ1 representative_theta_c1 #        array4
        # subQ1 representative_theta_c2 #        array5
        ##################################################
        n_representative_theta_c = len(representative_theta_c)
        subQ_id = int(i / n_representative_theta_c)
        c_id = representative_theta_c[int(i % n_representative_theta_c)]
        po_ind = is_pareto_efficient(sub_obj_values)
        po_objs = sub_obj_values[po_ind]
        po_configs = sub_config[po_ind, :]

        subQ_ids = (
            np.ones(
                [
                    po_objs.shape[0],
                ]
            )
            * subQ_id
        )
        c_ids = (
            np.ones(
                [
                    po_objs.shape[0],
                ]
            )
            * c_id
        )
        p_ids = np.arange(po_objs.shape[0])
        indices_arr = np.vstack((subQ_ids, c_ids, p_ids)).T
        return po_objs, po_configs, indices_arr

    @timeis
    def _optimize_theta_p_and_s(
        self,
        representative_theta_c: list,
        representative_theta_c_features: np.ndarray,
        theta_p_samples: np.ndarray,
        theta_s_samples: np.ndarray,
        n_subQs: int,
        use_ag: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """# fixme: to double-check
        optimize theta_p and theta_s (for simplicity, the following only mention
        theta_p as theta_p and theta_s function the same) under each fixed theta_c
        for all subQs
        :param representative_theta_c: the indices of representative theta_c in
                theta_c clusters
        :param representative_theta_c_features: the features of representative
                theta_c
        :param theta_p_samples: the initial theta_p samples
        :param theta_s_samples: the initial theta_s samples
        :param n_subQs: the number of subQs
        :param use_ag: flag to indicate whether using AutoGluon
        :return: optimal_obj_values_all_subQs: array, the optimal objective values
                        for all subQs
                optimal_configurations_all_subQs: the corresponding configurations,
                        where the length of columes is len_one_theta
                indices_all_subQs_theta_c_p: the array, where columes show subQ_ids,
                        theta_c_ids and ids of optimal theta_p solutions
                model_inference_arr: the array of model inference info, where
                        columes show [n_evals, model_inference_time]
                        e.g. n_evals is equal to the number of predictions/input
                        from the model
        """

        (
            n_evals,
            mesh_theta,
            mesh_graph_embeddings,
            mesh_non_decision_tabular_features,
        ), tc_construct_model_input_for_opt_p = self.construct_model_input_for_opt_p(
            representative_theta_c_features,
            theta_p_samples,
            theta_s_samples,
            n_subQs,
            self.use_ag,
            self.graph_embeddings,
            self.non_decision_tabular_features,
        )
        self.global_time_cost[self.construct_model_input_for_opt_p.__name__].append(
            tc_construct_model_input_for_opt_p
        )

        y_hat, model_inference_time = self.get_obj_values(
            n_evals,
            mesh_theta,
            mesh_graph_embeddings,
            mesh_non_decision_tabular_features,
        )
        self.global_time_cost[self.get_obj_values.__name__].append(model_inference_time)
        model_inference_arr = np.array([n_evals, model_inference_time])

        (
            optimal_obj_values_all_subQs,
            optimal_configurations_all_subQs,
            indices_all_subQs_theta_c_p,
        ), tc_filter_dominated_theta_p = self.filter_dominated_theta_p_s(
            representative_theta_c,
            theta_p_samples.shape[0],
            n_subQs,
            mesh_theta,
            y_hat,
            use_ag=use_ag,
        )
        self.global_time_cost[self.filter_dominated_theta_p_s.__name__].append(
            tc_filter_dominated_theta_p
        )

        return (
            optimal_obj_values_all_subQs,
            optimal_configurations_all_subQs,
            indices_all_subQs_theta_c_p,
            model_inference_arr,
        )

    @timeis
    def assign_optimal_theta_p_s(
        self,
        representative_theta_c: list,
        init_n_c_samples: int,
        cluster_members: list,
        theta_c_samples: np.ndarray,
        configurations_all_subQs: np.ndarray,
        indices_all_subQs: np.ndarray,
        n_subQs: int,
        use_ag: bool,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: Union[th.Tensor, pd.DataFrame],
        mode: str = "assign_initial",
    ) -> Tuple[
        Union[np.ndarray, th.Tensor],
        th.Tensor,
        Union[th.Tensor, pd.DataFrame],
        np.ndarray,
    ]:
        """todo: to refactor
        assign the optimal theta_p and theta_s to theta_c samples.
        :param representative_theta_c: the indices of representative theta_c in
                theta_c clusters
        :param cluster_members: the labels in each cluster
                                (e.g. 3 clusters, [np.array([0, 4, 5)],
                                 np.array([1, 2]), np.array([3])])
        :param theta_c_samples: theta_c samples, either generated from
                initialization or newly generated
        :param configurations_all_subQs: the configurations of all subQs, where the
                length of columes is len_one_theta
        :param indices_all_subQs: an array, where columes show subQ_ids,
                representative/initial theta_c_ids and ids of optimal theta_p
                solutions
        :param n_subQs: the number of subQs
        :param use_ag: flag to show whether using AutoGluon in predictive models
        :param graph_embeddings: the graph embeddings for all subQs
        :param non_decision_tabular_features: the non_decision_tabular_features for
                all subQs
        :param mode: to show whether to assign the optimal theta_p and theta_s to
                the initial theta_c_samples or newly generated theta_c, using
                "assign_initial" or "assign_new"
        :return: mesh_theta: configurations of all subQs and all theta_c, shape
                        (#solution, len_one_theta).
                mesh_graph_embedding: graph embeddings of all subQs and all theta_c,
                        shape (#solution, 32)
                mesh_non_decision_tabular_features: graph embeddings of all subQs
                        and all theta_c, shape (#solution, 28)
                indices_arr: indices of subQs, all theta_c and optimal theta_p
        """
        # col0: qs_id, col1: c_id, col2: p_id
        # to get the subQ_ids and theta_c_ids
        subQ_theta_c_inds_arr = indices_all_subQs[:, :2]

        # fixme: to use numpy directly rather transform it to tensor
        theta_c_features_tensor: th.Tensor
        if isinstance(theta_c_samples, np.ndarray):
            theta_c_features_tensor = th.tensor(theta_c_samples, dtype=th.float32)

        configurations_all_subQs_tensor: th.Tensor
        if isinstance(configurations_all_subQs, np.ndarray):
            configurations_all_subQs_tensor = th.tensor(
                configurations_all_subQs, dtype=th.float32
            )

        # list, the indices are the order of subQs
        optimal_theta_p_list = [
            # list, the indices are the order of theta_c_ids in representative
            # theta_c
            # each element is the configurations under a specific theta_c and subQ
            # each element is a tensor with shape (n_optimal_theta_p, len_one_theta)
            [
                configurations_all_subQs_tensor[
                    np.where((subQ_theta_c_inds_arr == [subQ_id, theta_c_id]).all(1))
                ].reshape(-1, self.len_theta)
                for theta_c_id in representative_theta_c
            ]
            for subQ_id in range(n_subQs)
        ]
        # len_opt_theta_p_per_subQ: list, the indices are the order of subQs
        # each element is a list, whose indices are the order of theta_c_ids in
        # representative theta_c
        # each element shows the number of optimal theta_p solutions under a fixed
        # theta_c and subQ
        len_opt_theta_p_per_subQ = [
            [x.shape[0] for x in opt_theta_p] for opt_theta_p in optimal_theta_p_list
        ]

        if mode == "assign_initial":
            # assign optimal theta_p solutions to all the initial theta_c samples
            # list follows the order of subQs, each element is a list
            replicate_theta_c_features = [
                # list follows the number of theta_c_ids in the representative
                # theta_c
                # each element is a tensor with shape (n_solutions, len_theta_c)
                # where n_solutions = n_cluster_members * n_optimal_theta_p
                # e.g. for subQ0, cluster0=[theta_c0, theta_c2], if optimizing
                # representative theta_c2 gets 1 optimal solution,
                # n_solutions = 2 * 1 = 2
                [
                    th.repeat_interleave(theta_c_features_tensor[cm], opt_p_len, dim=0)
                    for cm, opt_p_len in zip(cluster_members, x)
                ]
                for x in len_opt_theta_p_per_subQ
            ]
        else:
            # to assign optimal theta_p solutions to all the newly generated theta_c
            # samples
            replicate_theta_c_features = [
                [
                    th.repeat_interleave(
                        # settings which distinguishs the newly generated theta_c_ids,
                        # which is added with the number of initial theta_c samples
                        theta_c_features_tensor[
                            (np.array(cm) - init_n_c_samples).tolist()
                        ],
                        opt_p_len,
                        dim=0,
                    )
                    for cm, opt_p_len in zip(cluster_members, x)
                ]
                for x in len_opt_theta_p_per_subQ
            ]
        # concatenate to get one tensor of theta_c features for all subQs and
        # theta_c
        replicate_theta_c_features_all_subQs = th.cat(
            [th.cat(x) for x in replicate_theta_c_features]
        )

        # to replicate theta_p and theta_s features based on the number of optimal
        # theta_p solutions under all theta_c and subQs
        # list follows the order of subQs, each element is a list
        replicate_theta_p_s_features = [
            # list follows the theta_c_ids of representative theta_c,
            # each element is a tensor with shape (n_solutions, len_theta_p_s)
            # where n_solutions = n_cluster_members * n_optimal_theta_p
            # e.g. for subQ0, cluster0=[theta_c0, theta_c2], if optimizing
            # representative theta_c2 gets 1 optimal solution,
            # n_solutions = 2 * 1 = 2
            [
                opt_theta_p[:, theta_c_features_tensor.shape[1] :].repeat(len(cm), 1)
                for cm, opt_theta_p in zip(cluster_members, opt_theta_p_per_subQ)
            ]
            for opt_theta_p_per_subQ in optimal_theta_p_list
        ]
        # concatenate to get one tensor of theta_p_s features for all subQs and
        # theta_c
        replicate_theta_p_s_features_all_subQs = th.cat(
            [th.cat(x) for x in replicate_theta_p_s_features]
        )
        assert (
            replicate_theta_c_features_all_subQs.shape[0]
            == replicate_theta_p_s_features_all_subQs.shape[0]
        )
        mesh_theta = th.cat(
            [
                replicate_theta_c_features_all_subQs,
                replicate_theta_p_s_features_all_subQs,
            ],
            dim=1,
        )

        # mesh_theta_c_ids: to replicate all the theta_c_ids with its number of
        # optimal theta_p solutions
        # each element is a list, the indices are the order of subQs
        # note: to let the theta_c in the same cluster have the same optimal theta_p
        # solutions
        # e.g. given 2 representative theta_c (i.e. theta_c2, theta_c3),
        # i.e. 2 theta clusters
        # where theta_c2 in the cluster0 = [theta_c0, theta_c2],
        # theta_c3 in the cluster1 = [theta_c1, theta_c3]
        # theta_c2 has 1 and theta_c3 has 2 optimal theta_p solutions respectively
        # ##  mesh_theta_c_ids of each subQ (e.g. subQ0) #######
        #### theta_c0  #  theta_p0*   #
        #### theta_c2  #  theta_p0*   #
        #### theta_c1  #  theta_p1*   #
        #### theta_c1  #  theta_p2*   #
        #### theta_c3  #  theta_p1*   #
        #### theta_c3  #  theta_p2*   #
        ###############################
        replicate_theta_c_ids = [
            # list: it follows the order of theta_c_ids in the representative
            # theta_c
            # e.g. [theta_c2, theta_c3] in the above example
            # each element is a tensor with theta_c_ids replicated by its number of
            # optimal theta_p solutions, for all theta_c within the same cluster
            # e.g. theta_c2 has 1 and theta_c3 has 2 optimal theta_p solutions
            # [theta_c0, theta_c2], [theta_c1, theta_c1, theta_c3, theta_c3]
            [
                th.repeat_interleave(th.Tensor(cm), opt_p_len)
                for cm, opt_p_len in zip(cluster_members, x)
            ]
            for x in len_opt_theta_p_per_subQ
        ]
        # for each subQ, concatenate all replicated theta_c_ids of all clusters
        # list: it follows the order of subQs, each element is a tensor, with
        # shape(n_solutions, )
        # note: after concatenating, n_solutions could be different for different
        # subQs
        # e.g. given a query with 2 subQs, theta_c0, there are 1 optimal solution
        # in subQ0, and 2 in subQ1
        # replicate_theta_c_subQs would be [tensor(1, ), tensor(2, )]
        replicate_theta_c_subQs = [th.cat(x) for x in replicate_theta_c_ids]
        # replicate subQ_ids based on the number of solutions in
        # replicate_theta_c_subQs
        # The aim is to let subQ_ids align with the number of solutions in each subQ
        # list: follows the order of subQs, each element is a tensor
        replicate_subQ_ids = [
            th.Tensor([subQ_id]).repeat(x.shape[0])
            for subQ_id, x in zip(range(n_subQs), replicate_theta_c_subQs)
        ]
        mesh_graph_embedding = th.cat(
            # a list follows the order of subQs,
            # each element is a tensor with replicated graph embeddings for each subQ
            [
                graph_embeddings[subQ_id]
                .reshape(1, -1)
                .repeat_interleave(x.shape[0], dim=0)
                for subQ_id, x in zip(range(n_subQs), replicate_subQ_ids)
            ]
        )
        mesh_non_decision_tabular_features: Union[th.Tensor, pd.DataFrame]

        if use_ag:
            assert isinstance(non_decision_tabular_features, pd.DataFrame)
            mesh_theta = mesh_theta.numpy()
            mesh_non_decision_tabular_features = pd.concat(
                # a list follows the order of subQs,
                # each element is a pd.Dataframe with replicated non_decision
                # tabular features for each subQ
                [
                    non_decision_tabular_features.loc[
                        np.repeat(
                            non_decision_tabular_features.index[subQ_id],  # type: ignore
                            x.shape[0],
                        ).astype(int)
                    ].reset_index(drop=True)
                    for subQ_id, x in zip(range(n_subQs), replicate_subQ_ids)
                ]
            )
            assert (
                mesh_non_decision_tabular_features.shape[0]
                == mesh_graph_embedding.shape[0]
            )
        else:
            assert isinstance(non_decision_tabular_features, th.Tensor)
            mesh_non_decision_tabular_features = th.cat(
                [
                    non_decision_tabular_features[subQ_id]
                    .reshape(1, -1)
                    .repeat_interleave(x.shape[0], dim=0)
                    for subQ_id, x in zip(range(n_subQs), replicate_subQ_ids)
                ]
            )

        ########### update the array of indices (subQ_ids, theta_c_ids, and
        # optimal theta_p_s ids)
        replicate_theta_p_subQs = th.cat(
            [
                th.cat(
                    [
                        th.arange(opt_theta_p_len).repeat(len(cm))
                        for cm, opt_theta_p_len in zip(cluster_members, x)
                    ]
                )
                for x in len_opt_theta_p_per_subQ
            ]
        )
        indices_arr = np.vstack(
            [
                th.cat(replicate_subQ_ids).numpy(),
                th.cat(replicate_theta_c_subQs).numpy(),
                replicate_theta_p_subQs.numpy(),
            ]
        ).T

        return (
            mesh_theta,
            mesh_graph_embedding,
            mesh_non_decision_tabular_features,
            indices_arr,
        )

    @timeis
    def estimate_opt_theta_p_s(
        self,
        representative_theta_c: list,
        init_n_c_samples: int,
        cluster_members: list,
        theta_c_samples: np.ndarray,
        configurations_all_subQs: np.ndarray,
        indices_all_subQs: np.ndarray,
        n_subQs: int,
        use_ag: bool,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: Union[th.Tensor, pd.DataFrame],
        mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # fixme: to double-check
        """
        estimate the objective values of all theta_c using the optimal theta_p and
        theta_s of representative theta_c
        :param representative_theta_c: the indices of representative theta_c in
                theta_c clusters
        :param init_n_c_samples: the number of initial theta_c samples, used in
                assigning the optimal theta_p for the new theta_c samples,
                - as to distinguish new theta_c and the initial theta_c samples, the
                 indices of new theta_c sums the number of initial theta_c samples
        :param cluster_members: the labels in each cluster
                                (e.g. 3 clusters, [np.array([0, 4, 5)],
                                np.array([1, 2]), np.array([3])])
        :param theta_c_samples: theta_c samples, either generated from
                initialization or newly generated
        :param configurations_all_subQs: the configurations of all subQs, where the
                length of columes is len_one_theta
        :param indices_all_subQs: an array, where columes show subQ_ids, theta_c_ids
                and ids of optimal theta_p solutions
        :param n_subQs: the number of subQs
        :param use_ag: flag to show whether using AutoGluon in predictive models
        :param graph_embeddings: the graph embeddings for all subQs
        :param non_decision_tabular_features: the non_decision_tabular_features for
                all subQs
        :param mode: to show whether to assign the optimal theta_p and theta_s to
                the initial theta_c_samples or newly generated theta_c, using
                "assign_initial" or "assign_new"
        :return: obj_values_all_subQs: objective values of all_subQs and all_theta_c
                        with optimal theta_p and theta_s of representative theta_c,
                        shape (#solution, n_objs)
                mesh_theta: configurations of all subQs and all theta_c, shape
                        (#solution, len_one_theta)
                updated_indices_all_subQs: the indices of subQs, all theta_c and
                        the optimal theta_p
                model_inference_arr: the array of model inference info, where
                        columes show [n_evals, model_inference_time]
                        e.g. n_evals is equal to the number of predictions/input
                        from the model
        """

        (
            mesh_theta,
            mesh_graph_embedding,
            mesh_non_decision_tabular_features,
            updated_indices_all_subQs,
        ), tc_assign = self.assign_optimal_theta_p_s(
            representative_theta_c,
            init_n_c_samples,
            cluster_members,
            theta_c_samples,
            configurations_all_subQs,
            indices_all_subQs,
            n_subQs,
            use_ag,
            graph_embeddings,
            non_decision_tabular_features,
            mode=mode,
        )
        self.global_time_cost[self.assign_optimal_theta_p_s.__name__].append(tc_assign)

        n_eval = mesh_theta.shape[0]
        obj_values_all_subQs, tc_model_inference = self.get_obj_values(
            n_eval,
            mesh_theta,
            mesh_graph_embedding,
            mesh_non_decision_tabular_features,
        )
        self.global_time_cost[self.get_obj_values.__name__].append(tc_model_inference)
        model_inference_arr = np.array([n_eval, tc_model_inference])

        return (
            obj_values_all_subQs,
            mesh_theta,
            updated_indices_all_subQs,
            model_inference_arr,
        )

    @timeis
    def union_opt_theta_c(
        self,
        n_subQs: int,
        len_theta_c: int,
        objective_values_all_subQs: np.ndarray,
        configurations_all_subQs: np.ndarray,
        indices_all_subQs_all_theta_c: np.ndarray,
    ) -> List[List[Any]]:
        # fixme: to double-check
        """
        union the optimal theta_c (i.e. from the subQ-level solutions without
        constraints of identical theta_c) of all subQs to provide a warm start to
        generate new theta_c
        :param n_subQs: the number of subQs
        :param len_theta_c: the length of the theta_c, i.e. 8
        :param objective_values_all_subQs: objective values of all_subQs and
                all_theta_c with optimal theta_p and theta_s of representative
                theta_c, shape (#solution, n_objs)
        :param configurations_all_subQs: configurations of all subQs and all
                theta_c, shape (#solution, len_one_theta)
        :param indices_all_subQs_all_theta_c: the indices of subQs, all theta_c and
                the optimal theta_p
        :return: the optimal theta_c from all subQs
        """

        local_opt_theta_c_list = []
        for subQ_id in range(n_subQs):
            subQ_values_inds = np.where(indices_all_subQs_all_theta_c[:, 0] == subQ_id)[
                0
            ].tolist()
            subQ_values = objective_values_all_subQs[subQ_values_inds]
            po_ind = is_pareto_efficient(subQ_values)
            po_theta_c = configurations_all_subQs[subQ_values_inds][
                po_ind, :len_theta_c
            ].tolist()
            local_opt_theta_c_list.append(po_theta_c)

        union_opt_theta_c_list: List[List[Any]]
        union_opt_theta_c_list = sum(local_opt_theta_c_list, [])
        uniq_union_opt_theta_c_list = np.unique(union_opt_theta_c_list, axis=0).tolist()
        return uniq_union_opt_theta_c_list

    @timeis
    def extend_c(
        self,
        cross_location: int,
        union_opt_theta_c_list: list,
        use_ag: bool,
        theta_sample_mode: str,
        theta_c_samples: np.ndarray,
    ) -> List[List[Any]]:
        """
        Generate new theta_c samples
        :param cross_location: the location of crossover, e.g. the following is the
                case of cross_location=3
                ###############|##########################
                # k1 # k2 # k3 |# k4 # k5 # k6 # k7 # k8 #
                # k1'# k2'# k3'|# k4'# k5'# k6'# k7'# k8'#
                ###############|##########################
        :param union_opt_theta_c_list: the optimal theta_c from all subQs
        :param use_ag: flag to show whether using AutoGluon in predictive models
        :param theta_sample_mode: the mode to indicate how to get the initial theta
                samples, e.g. "grid-search", "random"
        :param theta_c_samples: theta_c samples
        :return: newly generated theta_c samples
        """
        if theta_sample_mode.startswith("grid"):
            (
                new_theta_c_list,
                tc_random_sample_new_theta_c,
            ) = self.random_samples_new_theta_c(union_opt_theta_c_list, use_ag)
        elif theta_sample_mode in ["random", "lhs"]:
            new_theta_c_list, tc_crossover = self.crossover(
                cross_location,
                union_opt_theta_c_list,
                theta_c_samples,
            )
        else:
            raise Exception(
                f"Theta sampling mode {theta_sample_mode} is not supported!"
            )

        return new_theta_c_list

    @timeis
    def crossover(
        self,
        cross_location: int,
        union_opt_theta_c_list: list,
        theta_c_samples: np.ndarray,
    ) -> List[List[Any]]:
        """
        crossover to generate new \theta_c
        :param cross_location: the location of crossover, e.g. the following is
                the case of cross_location=3
                ###############|##########################
                # k1 # k2 # k3 |# k4 # k5 # k6 # k7 # k8 #
                # k1'# k2'# k3'|# k4'# k5'# k6'# k7'# k8'#
                ###############|##########################
        :param union_opt_theta_c_list: the optimal theta_c from all subQs
        :param theta_c_samples: theta_c_samples
        :return: the newly generated theta_c, e.g.
        Given:
        ###############|##########################
        # k1 # k2 # k3 |# k4 # k5 # k6 # k7 # k8 #
        # k1'# k2'# k3'|# k4'# k5'# k6'# k7'# k8'#
        ###############|##########################
        After crossover
        ###############|##########################
        # k1 # k2 # k3 |# k4'# k5'# k6'# k7'# k8'#
        # k1'# k2'# k3'|# k4 # k5 # k6 # k7 # k8 #
        ###############|##########################
        """
        # the length of theta_c, i.e. 8
        len_theta_c = theta_c_samples.shape[1]
        uniq_res_c = np.unique(
            np.array(union_opt_theta_c_list)[:, :cross_location], axis=0
        )
        uniq_non_res_c = np.unique(
            np.array(union_opt_theta_c_list)[:, cross_location:len_theta_c], axis=0
        )

        if uniq_res_c.shape[0] <= uniq_non_res_c.shape[0]:
            theta_res_mesh = np.repeat(uniq_res_c, uniq_non_res_c.shape[0], axis=0)
            theta_c_non_res_mesh = np.tile(uniq_non_res_c, (uniq_res_c.shape[0], 1))
            extend_theta_c = np.hstack((theta_res_mesh, theta_c_non_res_mesh))
        else:
            theta_res_mesh = np.tile(uniq_res_c, (uniq_non_res_c.shape[0], 1))
            theta_c_non_res_mesh = np.repeat(
                uniq_non_res_c, uniq_res_c.shape[0], axis=0
            )
            extend_theta_c = np.hstack((theta_res_mesh, theta_c_non_res_mesh))

        # filter theta_c from the newly generated theta_c candidates,
        # which are already included in the union optimal theta_c candidates
        uniq_new_c = np.unique(extend_theta_c, axis=0)
        matching_rows_opt_c = np.all(
            uniq_new_c[:, None, :] == np.array(union_opt_theta_c_list)[None, :, :],
            axis=-1,
        )
        matching_indices_opt_c = np.where(matching_rows_opt_c.any(axis=-1))[0]
        all_new_c_inds = np.arange(uniq_new_c.shape[0])
        mask_opt_c = ~np.isin(all_new_c_inds, matching_indices_opt_c)
        filtered_inds_opt_c = all_new_c_inds[mask_opt_c]

        extend_c_filter_dup_opt_c = extend_theta_c[filtered_inds_opt_c]

        # filter theta_c from the newly generated theta_c candidates,
        # which are already included in the initial theta_c candidates,
        # i.e. clustering features
        if isinstance(theta_c_samples, np.ndarray):
            matching_rows_init_c = np.all(
                extend_c_filter_dup_opt_c[:, None, :] == theta_c_samples[None, :, :],
                axis=-1,
            )
        else:
            assert isinstance(theta_c_samples, th.Tensor)
            matching_rows_init_c = np.all(
                extend_c_filter_dup_opt_c[:, None, :]
                == theta_c_samples.numpy()[None, :, :],
                axis=-1,
            )
        matching_indices_init_c = np.where(matching_rows_init_c.any(axis=-1))[0]
        all_new_c_inds_init_c = np.arange(extend_c_filter_dup_opt_c.shape[0])
        mask_init_c = ~np.isin(all_new_c_inds_init_c, matching_indices_init_c)
        filtered_inds_init_c = all_new_c_inds_init_c[mask_init_c]

        extend_c_list = extend_c_filter_dup_opt_c[filtered_inds_init_c].tolist()
        return extend_c_list

    @timeis
    def random_samples_new_theta_c(
        self,
        union_opt_theta_c_list: list,
        use_ag: bool,
    ) -> List[List[Any]]:
        """
        Randomly generate new theta_c
        :param union_opt_theta_c_list: the optimal theta_c from all subQs
        :param use_ag: flag to indicate whether using AutoGluon in models
        :return: newly generated theta_c
        """
        n_new_samples = len(union_opt_theta_c_list)
        new_theta_c_list = self.theta_sample_funcs(
            n_samples=n_new_samples,
            theta_type="c",
            seed=self.seed + n_new_samples if self.seed is not None else None,
            selected_features=None,
            normalize=not use_ag,
            mode="random",
        ).tolist()
        return new_theta_c_list

    @timeis
    def assign_new_theta_c_to_clusters(
        self,
        new_theta_c_list: list,
        theta_c_samples: np.ndarray,
        representative_theta_c: list,
        cluster_model: object,
    ) -> Tuple[List[Any], List[int]]:
        """fixme: to double-check.
        assign the new theta_c to existing cluster labels.
        :param new_theta_c_list: newly generated theta_c samples
        :param theta_c_samples: initial theta_c samples
        :param representative_theta_c: the features of representative theta_c
        :param cluster_model: cluster models, which takes theta_c samples as input,
                and returns cluster labels
        :return: new_cluster_members: labels of theta_c clusters,
                new_representative_theta_c: representative_theta_c of the new
                        cluster labels
        """
        # predict cluster labels
        pred_cluster_label = cluster_model.predict(new_theta_c_list)  # type: ignore
        uniq_cluster_label = np.unique(pred_cluster_label)
        # colume 1: cluster id; colume 2: new theta_c id
        new_label_arr = np.vstack(
            (
                pred_cluster_label,
                np.arange(pred_cluster_label.shape[0]) + theta_c_samples.shape[0],
            )
        ).T
        new_cluster_members = [
            new_label_arr[:, 1][np.where(new_label_arr[:, 0] == label)].tolist()
            for label in uniq_cluster_label
        ]
        new_representative_theta_c = [
            representative_theta_c[k] for k in uniq_cluster_label
        ]

        return new_cluster_members, new_representative_theta_c

    @timeis
    def union_initial_and_new_solution_sets(
        self,
        obj_values_all_subQs_init_theta_c: np.ndarray,
        obj_values_all_subQs_new_theta_c: np.ndarray,
        config_all_subQs_init_theta_c: Union[th.Tensor, np.ndarray],
        config_all_subQs_new_theta_c: Union[th.Tensor, np.ndarray],
        indices_all_subQs_init_theta_c: np.ndarray,
        indices_all_subQs_new_theta_c: np.ndarray,
        model_inference_info_optimize_theta_p: np.ndarray,
        model_inference_info_assign_init_theta_c: np.ndarray,
        model_inference_info_assign_new_theta_c: np.ndarray,
    ) -> Tuple[np.ndarray, Union[th.Tensor, np.ndarray], np.ndarray, np.ndarray]:
        """
        Union the solutions sets from initial theta_c samples and newly generated
        theta_c_samples
        :param obj_values_all_subQs_init_theta_c: the objective values of all subQs
                from the initial theta_c samples
        :param obj_values_all_subQs_new_theta_c: the objective values of all subQs
                from the newly generated theta_c
        :param config_all_subQs_init_theta_c: the configurations of all subQs from
                the initial theta_c samples
        :param config_all_subQs_new_theta_c: the configurations of all subQs from
                the newly generated theta_c
        :param indices_all_subQs_init_theta_c: the indices of subQs, initial theta_c
                samples and optimal theta_p.
        :param indices_all_subQs_new_theta_c: the indices of subQs, new theta_c
                samples and optimal theta_p.
        :param model_inference_info_optimize_theta_p: the model inference information
                (n_evaluation, time cost) in optimizing theta_p and theta_s for each
                 theta_c of each subQ
        :param model_inference_info_assign_init_theta_c: the model inference
                information (n_evaluation, time cost) in assigning optimal theta_p
                and theta_s to each initial theta_c of each subQ
        :param model_inference_info_assign_new_theta_c: the model inference
                information (n_evaluation, time cost) in assigning optimal theta_p
                and theta_s to each new theta_c of each subQ
        :return: obj_values_all_subQs: the subQ-level Pareto optimal solutions,
                        which are Pareto optimal under each theta_c and subQ, shape
                         (#total_solutions, n_objs), here n_objs=2 with latency and
                          cost.
                config_all_subQs: the corresponding configurations, shape
                        (#total_solutions, len_one_theta), here
                        len_one_theta = len_theta_c + len_theta_p + len_theta_s = 19
                indices_all_subQs: the indices of subQs, theta_c and optimal theta_p
                        for the obj_valeus_all_subQs, with shape (#total_solution, 3),
                            e.g. np.array([[0, 1, 2]]) indicates the second
                            solution of subQ0 under theta_c_1 (indexing from 0).
                final_model_inference_info: shape (3, 2), where 3 means HMOOC calls
                        the predictive model 3 times; 2 shows two columes
                        "n_evaluations" and "inference_time" respectively,
                        - e.g. when 100 * len_one_theta is fed in the model,
                        n_evaluations = 100
        """
        config_all_subQs: Union[th.Tensor, np.ndarray]
        if isinstance(config_all_subQs_init_theta_c, np.ndarray) and isinstance(
            config_all_subQs_new_theta_c, np.ndarray
        ):
            config_all_subQs = np.concatenate(
                [config_all_subQs_init_theta_c, config_all_subQs_new_theta_c]
            )
        elif isinstance(config_all_subQs_init_theta_c, th.Tensor) and isinstance(
            config_all_subQs_new_theta_c, th.Tensor
        ):
            config_all_subQs = th.concatenate(
                [config_all_subQs_init_theta_c, config_all_subQs_new_theta_c]
            )
        else:
            raise TypeError(config_all_subQs_init_theta_c, config_all_subQs_new_theta_c)

        obj_values_all_subQs = np.concatenate(
            [obj_values_all_subQs_init_theta_c, obj_values_all_subQs_new_theta_c]
        )
        indices_all_subQs = np.concatenate(
            [indices_all_subQs_init_theta_c, indices_all_subQs_new_theta_c]
        )

        final_model_inference_info = np.vstack(
            (
                model_inference_info_optimize_theta_p,
                model_inference_info_assign_init_theta_c,
                model_inference_info_assign_new_theta_c,
            )
        )

        return (
            obj_values_all_subQs,
            config_all_subQs,
            indices_all_subQs,
            final_model_inference_info,
        )
