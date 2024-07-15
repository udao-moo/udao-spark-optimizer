from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch as th
from sklearn.preprocessing import MinMaxScaler
from udao.data import (
    NormalizePreprocessor,
    PredicateEmbeddingExtractor,
    QueryPlanIterator,
    TabularFeatureExtractor,
)
from udao.data.handler.data_processor import (
    DataProcessor,
    FeaturePipeline,
    create_data_processor,
)
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.preprocessors.normalize_preprocessor import FitTransformProtocol

from udao_trace.configuration import SparkConf

from ...utils.collaborators import TypeAdvisor
from ..extractors.query_structure_extractor import (
    QueryStructureExtractor2,
    get_extract_operations_from_serialized_json,
)


def create_udao_data_processor(
    ta: TypeAdvisor,
    sc: SparkConf,
    lpe_size: int,
    vec_size: int,
    tensor_dtypes: th.dtype = th.float32,
) -> DataProcessor:
    data_processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    tabular_columns = ta.get_tabular_columns()
    tabular_columns_scaler = MinMaxScalerWithTrainedTheta(tabular_columns, sc)
    return data_processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=tabular_columns),
            preprocessors=[NormalizePreprocessor(tabular_columns_scaler)],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=ta.get_objectives()),
        ),
        query_structure=FeaturePipeline(
            extractor=QueryStructureExtractor2(
                ta=ta, positional_encoding_size=lpe_size
            ),
            preprocessors=[NormalizePreprocessor(MinMaxScaler(), "graph_features")],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=vec_size)),
                extract_operations=get_extract_operations_from_serialized_json(
                    ta.q_type
                ),
            ),
        ),
    )


class MinMaxScalerWithTrainedTheta(FitTransformProtocol):
    def __init__(self, tabular_columns: List[str], sc: SparkConf):
        knob_selected_indices = [
            i for i, k in enumerate(sc.knob_ids) if k in tabular_columns
        ]
        knob_cols = np.array(sc.knob_ids)[knob_selected_indices]
        knob_scaler = MinMaxScaler()
        knob_scaler.fit(
            pd.DataFrame(
                [
                    np.array(sc.knob_min)[knob_selected_indices],
                    np.array(sc.knob_max)[knob_selected_indices],
                ],
                columns=knob_cols,
            )
        )
        self.theta_cols = knob_cols
        self.theta_scaler = knob_scaler
        self.non_theta_cols: Optional[List[str]] = None
        self.non_theta_scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MinMaxScalerWithTrainedTheta":
        if self.non_theta_cols is None:
            self.non_theta_cols = [c for c in X.columns if c not in self.theta_cols]
            if (len(self.theta_cols) + len(self.non_theta_cols)) != len(X.columns):
                raise Exception("theta_cols + non_theta_cols != all_cols")
        self.non_theta_scaler.fit(X[self.non_theta_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.non_theta_cols is None:
            raise Exception("fit() must be called before transform()")
        X[self.theta_cols] = self.theta_scaler.transform(X[self.theta_cols])
        X[self.non_theta_cols] = self.non_theta_scaler.transform(X[self.non_theta_cols])
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.non_theta_scaler is None:
            raise Exception("fit() must be called before inverse_transform()")
        X[self.theta_cols] = self.theta_scaler.inverse_transform(X[self.theta_cols])
        X[self.non_theta_cols] = self.non_theta_scaler.inverse_transform(
            X[self.non_theta_cols]
        )
        return X
