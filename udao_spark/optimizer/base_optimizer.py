import json
import os.path
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch as th
from scipy.stats import qmc
from udao.data.handler.data_processor import DataProcessor
from udao.data.iterators.query_plan_iterator import QueryPlanInput
from udao.data.predicate_embedders.utils import prepare_operation
from udao.data.utils.query_plan import add_positional_encoding
from udao.model.embedders.layers.multi_head_attention import RAALMultiHeadAttentionLayer
from udao.optimization.utils.moo_utils import Point, get_default_device

from udao_trace.configuration import SparkConf
from udao_trace.utils import PickleHandler

from ..data.extractors.query_structure_extractor import (
    extract_query_plan_features_from_serialized_json,
)
from ..model.embedders.graph_transformer import GraphTransformer
from ..model.model_server import AGServer
from ..model.utils import add_dist_to_graph, get_non_siblings, get_non_siblings_map
from ..utils.constants import THETA_C, THETA_COMPILE, THETA_P, THETA_S
from ..utils.logging import logger
from ..utils.monitor import UdaoMonitor
from ..utils.params import QType
from .utils import get_cloud_cost_add_io, get_cloud_cost_wo_io

ThetaType = Literal["c", "p", "s"]


class BaseOptimizer(ABC):
    def __init__(
        self,
        bm: str,
        model_sign: str,
        graph_model_params_path: str,
        graph_weights_path: str,
        q_type: QType,
        data_processor_path: str,
        spark_conf: SparkConf,
        decision_variables: List[str],
        ag_path: str,
        clf_json_path: Optional[str],
        clf_recall_xhold: float,
        verbose: bool = False,
    ) -> None:
        data_processor = PickleHandler.load(
            os.path.dirname(data_processor_path), os.path.basename(data_processor_path)
        )
        if not isinstance(data_processor, DataProcessor):
            raise TypeError(f"Expected DataProcessor, got {type(data_processor)}")
        if (
            "tabular_features" not in data_processor.feature_extractors
            or "objectives" not in data_processor.feature_extractors
        ):
            raise ValueError(
                "DataProcessor must contain tabular_features and objectives"
            )
        self.data_processor = data_processor
        feature_extractors = self.data_processor.feature_extractors
        feature_processors = self.data_processor.feature_processors

        self.query_structure = self.data_processor.feature_extractors["query_structure"]
        self.query_normalizer_cbo = feature_processors["query_structure"][0].normalizer
        self.op_enc_extractor = feature_extractors["op_enc"]

        self.tabular_columns = feature_extractors["tabular_features"].columns
        self.tabular_normalizer = feature_processors["tabular_features"][0].normalizer
        self.tabular_normalizer_meta = {
            "theta_cols": self.tabular_normalizer.theta_cols,
            "theta_min": self.tabular_normalizer.theta_scaler.data_min_,
            "theta_max": self.tabular_normalizer.theta_scaler.data_max_,
            "non_theta_cols": self.tabular_normalizer.non_theta_cols,
            "non_theta_min": self.tabular_normalizer.non_theta_scaler.data_min_,
            "non_theta_max": self.tabular_normalizer.non_theta_scaler.data_max_,
        }

        self.model_objective_columns = feature_extractors["objectives"].columns
        self.sc = spark_conf
        if decision_variables != self.tabular_columns[-len(decision_variables) :]:
            raise ValueError(
                "Decision variables must be the last columns in tabular_features"
            )
        if not all(v in THETA_COMPILE for v in decision_variables):
            raise ValueError(
                f"Decision variables must be in {THETA_COMPILE}, "
                f"got {decision_variables}"
            )
        self.decision_variables = decision_variables
        self.dtype = th.float32
        self.device = get_default_device()

        self.theta_all_minmax = (
            np.array(spark_conf.knob_min),
            np.array(spark_conf.knob_max),
        )
        self.theta_minmax = {
            "c": (
                np.array(spark_conf.knob_min[: len(THETA_C)]),
                np.array(spark_conf.knob_max[: len(THETA_C)]),
            ),
            "p": (
                np.array(
                    spark_conf.knob_min[len(THETA_C) : len(THETA_C) + len(THETA_P)]
                ),
                np.array(
                    spark_conf.knob_max[len(THETA_C) : len(THETA_C) + len(THETA_P)]
                ),
            ),
            "s": (
                np.array(spark_conf.knob_min[-len(THETA_S) :]),
                np.array(spark_conf.knob_max[-len(THETA_S) :]),
            ),
        }
        self.theta_names = {
            "c": THETA_C,
            "p": THETA_P,
            "s": THETA_S,
        }
        self.theta_default_all = spark_conf.deconstruct_configuration(
            np.array([[k.default for k in spark_conf.knob_list]])
        )
        self.theta_default = {
            "c": self.theta_default_all[:, : len(THETA_C)],
            "p": self.theta_default_all[:, len(THETA_C) : len(THETA_C) + len(THETA_P)],
            "s": self.theta_default_all[:, -len(THETA_S) :],
        }
        self.theta_ktype = {
            "c": [k.ktype for k in spark_conf.knob_list[: len(THETA_C)]],
            "p": [
                k.ktype
                for k in spark_conf.knob_list[
                    len(THETA_C) : len(THETA_C) + len(THETA_P)
                ]
            ],
            "s": [k.ktype for k in spark_conf.knob_list[-len(THETA_S) :]],
        }
        self.bm = bm
        self.ag_ms = AGServer.from_ckp_path(
            model_sign,
            graph_model_params_path,
            graph_weights_path,
            q_type,
            ag_path,
            clf_json_path,
            clf_recall_xhold,
        )
        self.ta = self.ag_ms.ta
        self.verbose = verbose

        if model_sign == "graph_raal":
            embedder = self.ag_ms.ms.model.embedder
            if not isinstance(embedder, GraphTransformer):
                raise ValueError(f"Expected GraphTransformer, got {type(embedder)}")
            template_plans = data_processor.feature_extractors[
                "query_structure"
            ].template_plans
            non_siblings_map = get_non_siblings_map(
                {k: query.graph for k, query in template_plans.items()}
            )
            for layer in embedder.layers:
                layer.non_siblings_map = non_siblings_map
                if not isinstance(layer.attention, RAALMultiHeadAttentionLayer):
                    raise ValueError(
                        f"Expected RAALMultiHeadAttentionLayer, got {type(layer)}"
                    )
                layer.attention.non_siblings_map = non_siblings_map

    def fast_extraction(
        self, df: pd.DataFrame, use_ag: bool
    ) -> Tuple[dgl.DGLGraph, Optional[th.Tensor]]:
        """not general but fast for our case"""
        bg = []
        for j in df[self.ta.get_graph_column()].values:
            d = json.loads(j)
            dt1 = time.perf_counter_ns()
            structure, op_features = extract_query_plan_features_from_serialized_json(
                self.ta, j
            )
            dt2 = time.perf_counter_ns()
            if self.verbose:
                logger.info(
                    f"~~~~ extract_query_plan_features_from_serialized_json "
                    f"in {(dt2 - dt1) / 1e6} ms"
                )
            op_gid = [
                self.query_structure.operation_types.get(n.split()[0])
                for n in structure.node_id2name.values()
            ]
            g = structure.graph
            g.ndata["op_gid"] = th.tensor(op_gid, dtype=th.int32)
            cbo_df = pd.DataFrame.from_dict(op_features.features_dict, dtype="float32")
            g.ndata["cbo"] = th.tensor(
                self.query_normalizer_cbo.transform(cbo_df), dtype=self.dtype
            )
            g.ndata["op_enc"] = th.tensor(
                self.op_enc_extractor.embedder.transform(
                    [
                        prepare_operation(op["predicate"])
                        for op in d["operators"].values()
                    ]
                ),
                dtype=self.dtype,
            )
            dt3 = time.perf_counter_ns()
            if self.verbose:
                logger.info(f"~~~~ prepare_operation in {(dt3 - dt2) / 1e6} ms")
            # if self.ag_ms.ms.model_sign == "graph_gtn":
            g = add_positional_encoding(
                g, self.query_structure.positional_encoding_size
            )
            dt4 = time.perf_counter_ns()
            if self.verbose:
                logger.info(f"~~~~ add_positional_encoding in {(dt4 - dt3) / 1e6} ms")
            bg.append(g)

        dt5 = time.perf_counter_ns()
        embedding_input = dgl.batch(bg)
        dt6 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f"~~~~ dgl.batch in {(dt6 - dt5) / 1e6} ms")

        if not use_ag:
            norm_non_theta = (
                df[self.tabular_normalizer_meta["non_theta_cols"]].values
                - self.tabular_normalizer_meta["non_theta_min"]
            ) / (
                self.tabular_normalizer_meta["non_theta_max"]
                - self.tabular_normalizer_meta["non_theta_min"]
            )
            norm_theta = (
                df[self.tabular_normalizer_meta["theta_cols"]].values
                - self.tabular_normalizer_meta["theta_min"]
            ) / (
                self.tabular_normalizer_meta["theta_max"]
                - self.tabular_normalizer_meta["theta_min"]
            )
            tabular_input = np.concatenate([norm_non_theta, norm_theta], axis=1)
            tabular_input = th.tensor(tabular_input, dtype=self.dtype)
            return embedding_input, tabular_input
        else:
            return embedding_input, None

    def general_extraction(
        self, df: pd.DataFrame
    ) -> Tuple[dgl.DGLGraph, Optional[th.Tensor]]:
        t1 = time.perf_counter_ns()
        with th.no_grad():
            iterator = self.data_processor.make_iterator(df, df.index, split="test")
        t2 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> created iterator in {(t2 - t1) / 1e6} ms")

        n_items = len(df)
        dataloader = iterator.get_dataloader(batch_size=n_items)
        batch_input, _ = next(iter(dataloader))
        if not isinstance(batch_input, QueryPlanInput):
            raise TypeError(f"Expected QueryPlanInput, got {type(batch_input)}")

        t3 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> created dataloader in {(t3 - t2) / 1e6} ms")

        embedding_input = batch_input.embedding_input
        tabular_input = batch_input.features
        return embedding_input, tabular_input

    def extract_non_decision_embeddings_from_df(
        self,
        df: pd.DataFrame,
        use_ag: bool = True,
        ercilla: bool = True,
        graph_choice: str = "gtn",
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, float]]:
        """
        compute the graph_embedding and
        the normalized values of the non-decision variables
        """

        t1 = time.perf_counter_ns()

        if df.index.name != "id":
            raise ValueError(">>> df must have an index named 'id'")

        df[self.decision_variables] = 0.0
        df[self.model_objective_columns] = 0.0

        t2 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> preprocessed df in {(t2 - t1) / 1e6} ms")

        if ercilla and graph_choice != "raal":
            embedding_input, tabular_input = self.fast_extraction(df, use_ag)
        else:
            embedding_input, tabular_input = self.general_extraction(df)

        # add dist to graph for QF
        if graph_choice == "qf":
            embedding_input = add_dist_to_graph(embedding_input)
        elif graph_choice == "raal":
            # add to simulate the computational cost
            get_non_siblings(embedding_input)

        t3 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> fast_extraction in {(t3 - t2) / 1e6} ms")

        graph_embedding = self.ag_ms.ms.model.embedder(embedding_input.to(self.device))

        if not use_ag:
            if tabular_input is None:
                raise ValueError("tabular_input must not be None")
            non_decision_tabular_features = tabular_input[
                :, : -len(self.decision_variables)
            ]
        else:
            non_decision_tabular_features = th.zeros(1, 1, dtype=self.dtype)

        t4 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">>> computed graph_embedding in {(t4 - t3) / 1e6} ms")

        return (
            graph_embedding,
            non_decision_tabular_features,
            {
                "input_extraction_ms": (t3 - t1) / 1e6,
                "graph_embedding_ms": (t4 - t3) / 1e6,
            },
        )

    @abstractmethod
    def extract_non_decision_df(self, non_decision_input: Dict) -> pd.DataFrame:
        ...

    def extract_data_and_compute_non_decision_features(
        self,
        monitor: UdaoMonitor,
        non_decision_input: Dict[str, Any],
        use_ag: bool = True,
        ercilla: bool = True,
        graph_choice: str = "gtn",
    ) -> Dict[str, Any]:
        t1 = time.perf_counter_ns()
        non_decision_df = self.extract_non_decision_df(non_decision_input)
        t2 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f">> extracted non_decision_df in {(t2 - t1) / 1e6} ms")
        (
            graph_embeddings,
            non_decision_tabular_features,
            time_dict,
        ) = self.extract_non_decision_embeddings_from_df(
            non_decision_df, use_ag=use_ag, ercilla=ercilla, graph_choice=graph_choice
        )
        t3 = time.perf_counter_ns()
        # add time measurements to
        monitor.input_extraction_ms += (t2 - t1) / 1e6  # monitoring
        monitor.input_extraction_ms += time_dict["input_extraction_ms"]  # monitoring
        monitor.graph_embedding_computation_ms = time_dict[
            "graph_embedding_ms"
        ]  # monitoring

        if self.verbose:
            logger.info(f">> extracted non_decision_embeddings in {(t3 - t2) / 1e6} ms")
            logger.debug("graph_embeddings shape: %s", graph_embeddings.shape)
            logger.warning(
                "non_decision_tabular_features is only used for "
                "MLP inference, shape: %s",
                non_decision_tabular_features.shape,
            )

        return {
            "non_decision_df": non_decision_df,
            "graph_embeddings": graph_embeddings,
            "non_decision_tabular_features": non_decision_tabular_features,
        }

    def summarize_obj(
        self,
        k1: np.ndarray,
        k2: np.ndarray,
        k3: np.ndarray,
        obj_lat: np.ndarray,  # lat or ana_lat
        obj_io: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        obj_cost_wo_io = get_cloud_cost_wo_io(
            lat=obj_lat,
            cores=k1,
            mem=k1 * k2,
            nexec=k3,
        )
        obj_cost_w_io = get_cloud_cost_add_io(obj_cost_wo_io, obj_io)
        if not isinstance(obj_cost_wo_io, np.ndarray) or not isinstance(
            obj_cost_w_io, np.ndarray
        ):
            raise TypeError(
                f"Expected np.ndarray, "
                f"got {type(obj_cost_wo_io)} and {type(obj_cost_w_io)}"
            )
        if "ana_latency_s" in self.ta.get_ag_objectives():
            return {
                "ana_latency": obj_lat,
                "io": obj_io,
                "ana_cost_wo_io": obj_cost_wo_io,
                "ana_cost_w_io": obj_cost_w_io,
            }
        else:
            return {
                "latency": obj_lat,
                "io": obj_io,
                "cost_wo_io": obj_cost_wo_io,
                "cost_w_io": obj_cost_w_io,
            }

    def get_latencies_and_objectives(
        self, objs_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if "ana_latency" in objs_dict:
            return objs_dict["ana_latency"], objs_dict["ana_cost_w_io"]
        else:
            return objs_dict["latency"], objs_dict["cost_w_io"]

    def predict_objectives_mlp(
        self, graph_embedding: th.Tensor, tabular_features: th.Tensor
    ) -> th.Tensor:
        """
        return multiple objective values for a given graph embedding and
        tabular features in the normalized space
        """
        return self.ag_ms.predict_with_mlp(
            graph_embedding, tabular_features.to(self.device)
        )

    def get_objective_values_mlp(
        self,
        graph_embeddings: th.Tensor,
        non_decision_tabular_features: th.Tensor,
        theta: th.Tensor,
    ) -> Dict[str, np.ndarray]:
        tabular_features = th.cat([non_decision_tabular_features, theta], dim=1)
        objs = self.predict_objectives_mlp(graph_embeddings, tabular_features).numpy()
        theta_c_min, theta_c_max = self.theta_minmax["c"]
        k1_min, k2_min, k3_min = theta_c_min[:3]
        k1_max, k2_max, k3_max = theta_c_max[:3]

        k1_norm = tabular_features[:, -len(THETA_C + THETA_P + THETA_S)].numpy()
        k2_norm = tabular_features[:, -len(THETA_C + THETA_P + THETA_S) + 1].numpy()
        k3_norm = tabular_features[:, -len(THETA_C + THETA_P + THETA_S) + 2].numpy()
        k1 = (k1_norm - k1_min) * (k1_max - k1_min) + k1_min
        k2 = (k2_norm - k2_min) * (k2_max - k2_min) + k2_min
        k3 = (k3_norm - k3_min) * (k3_max - k3_min) + k3_min

        if self.ta.q_type.startswith("qs_"):
            obj_io = objs[:, 1]
            obj_ana_lat = objs[:, 2]
            return self.summarize_obj(k1, k2, k3, obj_ana_lat, obj_io)
        else:
            obj_lat = objs[:, 0]
            obj_io = objs[:, 1]
            return self.summarize_obj(k1, k2, k3, obj_lat, obj_io)

    def get_objective_values_ag(
        self,
        template: str,
        graph_embeddings: np.ndarray,
        non_decision_df: pd.DataFrame,
        sampled_theta: np.ndarray,
        model_name: Dict[str, str],
    ) -> Dict[str, np.ndarray]:
        start_time_ns = time.perf_counter_ns()
        objs = self.ag_ms.predict_with_ag(
            self.bm,
            template,
            graph_embeddings,
            non_decision_df,
            self.decision_variables,
            sampled_theta,
            model_name,
        )
        end_time_ns = time.perf_counter_ns()
        logger.info(
            f"Takes {(end_time_ns - start_time_ns) / 1e6} ms "
            f"to compute {len(sampled_theta)} theta"
        )

        if "k1" in non_decision_df.columns:
            k1 = non_decision_df["k1"].values
            k2 = non_decision_df["k2"].values
            k3 = non_decision_df["k3"].values
        elif "k1" in self.decision_variables:
            if sampled_theta.ndim == 1:
                raise ValueError("sampled_theta must be 2D")
            k1 = sampled_theta[:, 0]
            k2 = sampled_theta[:, 1]
            k3 = sampled_theta[:, 2]
        else:
            raise ValueError("k1, k2, k3 not found")

        obj_lat = (
            objs["ana_latency_s"]
            if self.ta.q_type.startswith("qs_")
            else objs["latency_s"]
        )
        obj_io = objs["io_mb"]
        return self.summarize_obj(
            np.array(k1), np.array(k2), np.array(k3), obj_lat, obj_io
        )

    def sample_theta_all(
        self,
        n_samples: int,
        seed: Optional[int],
        selected_features: Optional[Dict[str, List[str]]],
        normalize: bool,
        mode: str = "random",
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        if selected_features is not None:
            theta_params = (
                self.theta_names["c"] + self.theta_names["p"] + self.theta_names["s"]
            )
            total_selected_features = (
                selected_features["c"] + selected_features["p"] + selected_features["s"]
            )
            selected_ids = []
            for c in total_selected_features:
                if c not in theta_params:
                    raise ValueError(f"{c} is not in {theta_params}")
                selected_ids.append(theta_params.index(c))
        else:
            selected_ids = list(range(len(self.theta_all_minmax[0])))

        theta_min = self.theta_all_minmax[0][selected_ids]
        theta_max = self.theta_all_minmax[1][selected_ids]
        n_params = len(selected_ids)
        samples = np.repeat(self.theta_default_all, n_samples, axis=0)
        if n_params == 0:
            logger.warning("no selected features, force to use default")
        else:
            if mode == "random":
                samples_selected = np.random.randint(
                    low=theta_min,  # inclusive
                    high=theta_max + 1,  # exclusive
                    size=(n_samples, n_params),
                )
            elif mode == "lhs":
                # a trivial implementation of Latin Hypercube Sampling when
                # all parameters(knobs) are integers
                sampler = qmc.LatinHypercube(d=n_params, seed=seed)
                samples_normed = sampler.random(n_samples)
                samples_selected = np.floor(
                    samples_normed * (theta_max + 1 - theta_min) + theta_min
                ).astype(int)
            else:
                raise ValueError(f"mode {mode} is not supported")
            samples[:, selected_ids] = samples_selected

        if normalize:
            samples_normalized = (samples - self.theta_all_minmax[0]) / (
                self.theta_all_minmax[1] - self.theta_all_minmax[0]
            )
            return samples_normalized
        else:
            return samples

    def sample_theta_x(
        self,
        n_samples: int,
        theta_type: ThetaType,
        seed: Optional[int],
        selected_features: Optional[Dict[str, List[str]]],
        normalize: bool = True,
        mode: str = "random",
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        if selected_features is not None:
            theta_params = self.theta_names[theta_type]
            selected_ids = []
            for c in selected_features[theta_type]:
                if c not in theta_params:
                    raise ValueError(f"{c} is not in {theta_params}")
                selected_ids.append(theta_params.index(c))
        else:
            selected_ids = list(range(len(self.theta_names[theta_type])))

        theta_min = self.theta_minmax[theta_type][0][selected_ids]
        theta_max = self.theta_minmax[theta_type][1][selected_ids]
        n_params = len(selected_ids)

        samples = np.repeat(self.theta_default[theta_type], n_samples, axis=0)
        if n_params == 0:
            logger.warning(
                f"no selected features for {theta_type}, force to use default"
            )
        else:
            if mode == "random":
                samples_selected = np.random.randint(
                    low=theta_min,
                    high=theta_max + 1,
                    size=(n_samples, n_params),
                )
            elif mode == "lhs":
                # a trivial implementation of Latin Hypercube Sampling when
                # all parameters(knobs) are integers
                sampler = qmc.LatinHypercube(d=n_params, seed=seed)
                samples_normed = sampler.random(n_samples)
                samples_selected = np.floor(
                    samples_normed * (theta_max + 1 - theta_min) + theta_min
                ).astype(int)
            else:
                raise ValueError(f"mode {mode} is not supported")
            samples[:, selected_ids] = samples_selected

        if normalize:
            samples_normalized = (samples - self.theta_minmax[theta_type][0]) / (
                self.theta_minmax[theta_type][1] - self.theta_minmax[theta_type][0]
            )
            return samples_normalized
        else:
            return samples

    def foo_samples(
        self, n_stages: int, seed: Optional[int], normalize: bool, mode: str = "random"
    ) -> np.ndarray:
        # a naive way to sample
        theta_c = np.tile(
            self.sample_theta_x(
                1,
                "c",
                seed if seed is not None else None,
                selected_features=None,
                normalize=normalize,
                mode=mode,
            ),
            (n_stages, 1),
        )
        theta_p = self.sample_theta_x(
            n_stages,
            "p",
            seed + 1 if seed is not None else None,
            selected_features=None,
            normalize=normalize,
            mode=mode,
        )
        theta_s = self.sample_theta_x(
            n_stages,
            "s",
            seed + 2 if seed is not None else None,
            selected_features=None,
            normalize=normalize,
            mode=mode,
        )
        theta = np.concatenate([theta_c, theta_p, theta_s], axis=1)
        return theta

    @abstractmethod
    def solve(
        self,
        template: str,
        non_decision_input: Dict[str, Any],
        seed: Optional[int] = None,
        use_ag: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        pass

    def weighted_utopia_nearest(self, pareto_points: List[Point]) -> Point:
        """
        return the Pareto point that is closest to the utopia point
        in a weighted distance function
        """
        # todo
        raise NotImplementedError
