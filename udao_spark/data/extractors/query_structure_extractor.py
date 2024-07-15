from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from udao.data import QueryStructureContainer, QueryStructureExtractor
from udao.data.predicate_embedders.utils import build_unique_operations
from udao.data.utils.query_plan import QueryPlanOperationFeatures, QueryPlanStructure
from udao.data.utils.utils import DatasetType

from udao_trace.utils import JsonHandler

from ...utils.collaborators import TypeAdvisor
from ...utils.exceptions import NoQTypeError
from ...utils.params import QType


class OperatorMisMatchError(BaseException):
    """raise when the operator names from `operator` and `link` do not match"""


def extract_operations_from_serialized_json_base(
    graph_column: str,
    plan_df: pd.DataFrame,
    operation_processing: Callable[[str], str] = lambda x: x,
) -> Tuple[Dict[int, List[int]], List[str]]:
    df = plan_df[["id", graph_column]].copy()
    df[graph_column] = df[graph_column].apply(
        lambda lqp_str: [
            operation_processing(op["predicate"])
            for op_id, op in JsonHandler.load_json_from_str(lqp_str)[
                "operators"
            ].items()
        ]  # type: ignore
    )
    df = df.explode(graph_column, ignore_index=True)
    df.rename(columns={graph_column: "operation"}, inplace=True)
    return build_unique_operations(df)


def extract_operations_from_serialized_lqp_json(
    plan_df: pd.DataFrame, operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    return extract_operations_from_serialized_json_base(
        "lqp", plan_df, operation_processing
    )


def extract_operations_from_serialized_qs_lqp_json(
    plan_df: pd.DataFrame, operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    return extract_operations_from_serialized_json_base(
        "qs_lqp", plan_df, operation_processing
    )


def extract_operations_from_serialized_qs_pqp_json(
    plan_df: pd.DataFrame, operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    return extract_operations_from_serialized_json_base(
        "qs_pqp", plan_df, operation_processing
    )


def get_extract_operations_from_serialized_json(q_type: QType) -> Callable:
    if q_type in ["q_compile", "q_all"]:
        return extract_operations_from_serialized_lqp_json
    if q_type in ["qs_lqp_compile", "qs_lqp_runtime"]:
        return extract_operations_from_serialized_qs_lqp_json
    if q_type in ["qs_pqp_runtime"]:
        return extract_operations_from_serialized_qs_pqp_json
    raise NoQTypeError(q_type)


def extract_query_plan_features_from_serialized_json(
    ta: TypeAdvisor,
    graph_json_str: str,
) -> Tuple[QueryPlanStructure, QueryPlanOperationFeatures]:
    graph = JsonHandler.load_json_from_str(graph_json_str)
    operators, links = graph["operators"], graph["links"]
    num_operators = len(operators)
    id2name = {int(op_id): ta.get_op_name(op) for op_id, op in operators.items()}
    incoming_ids: List[int] = []
    outgoing_ids: List[int] = []
    for link in links:
        from_id, to_id = link["fromId"], link["toId"]
        from_name, to_name = link["fromName"], link["toName"]
        if from_name != id2name[from_id]:
            id2name[from_id] = from_name
        if to_name != id2name[to_id]:
            id2name[to_id] = to_name
        incoming_ids.append(from_id)
        outgoing_ids.append(to_id)

    ranges = range(num_operators)
    node_names = [id2name[i] for i in ranges]
    sizes = [ta.size_mb_in_log(operators[str(i)]) for i in ranges]
    rows_count = [ta.rows_count_in_log(operators[str(i)]) for i in ranges]
    op_features = QueryPlanOperationFeatures(rows_count=rows_count, size=sizes)
    structure = QueryPlanStructure(
        node_names=node_names, incoming_ids=incoming_ids, outgoing_ids=outgoing_ids
    )
    return structure, op_features


class QueryStructureExtractor2(QueryStructureExtractor):
    """A uniformed graph extract serving for
    1. Query level logical query plan (q, q_compile)
    2. QueryStage level logical query plan with
        - estimated stats (qs_lqp)
        - actual stats (qs_lqp_oracle)
    3. QueryStage level physical query plan (qs_pqp)
    """

    def __init__(self, ta: TypeAdvisor, positional_encoding_size: Optional[int] = None):
        super(QueryStructureExtractor2, self).__init__(positional_encoding_size)
        self.ta = ta
        self.graph_column = ta.get_graph_column()

    def _extract_structure_and_features(
        self, idx: str, graph_json_str: str, split: DatasetType
    ) -> Dict:
        structure, op_features = extract_query_plan_features_from_serialized_json(
            self.ta, graph_json_str
        )
        operation_gids = self._extract_operation_types(structure, split)
        self.id_template_dict[idx] = self._extract_structure_template(structure, split)
        return {
            "operation_id": op_features.operation_ids,
            "operation_gid": operation_gids,
            **op_features.features_dict,
        }

    def extract_features(
        self, df: pd.DataFrame, split: DatasetType
    ) -> QueryStructureContainer:
        if self.graph_column not in df.columns:
            raise ValueError(f"graph_column {self.graph_column} not found in df")
        df_op_features: pd.DataFrame = df.apply(
            lambda row: self._extract_structure_and_features(
                row.id, row[self.graph_column], split
            ),
            axis=1,
        ).apply(pd.Series)
        df_op_features["plan_id"] = df["id"]
        (
            df_op_features_exploded,
            df_operation_types,
        ) = self._extract_op_features_exploded(df_op_features)
        return QueryStructureContainer(
            graph_features=df_op_features_exploded,
            template_plans=self.template_plans,
            key_to_template=self.id_template_dict,
            graph_meta_features=None,
            operation_types=df_operation_types,
        )
