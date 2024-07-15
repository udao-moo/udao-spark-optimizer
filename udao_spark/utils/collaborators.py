from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from udao_trace.utils import JsonHandler

from .constants import (
    ALPHA,
    ALPHA_COMPILE,
    ALPHA_QS_PLUS,
    BETA,
    EPS,
    GAMMA,
    THETA_C,
    THETA_P,
    THETA_S,
)
from .exceptions import NoBenchmarkError, NoQTypeError
from .logging import logger
from .params import ExtractParams, QType


def get_data_sign(bm: str, debug: bool) -> str:
    if bm == "tpch":
        return f"22x{10 if debug else 2273}"
    if bm == "tpcds":
        return f"102x{10 if debug else 490}"
    raise NoBenchmarkError


class PathWatcher:
    def __init__(
        self,
        base_dir: Path,
        benchmark: str,
        debug: bool,
        extract_params: ExtractParams,
        fold: Optional[int],
    ):
        self.base_dir = base_dir
        self.benchmark = benchmark
        self.debug = debug
        self.fold = fold

        data_sign = self._get_data_sign()
        data_prefix = f"{str(base_dir)}/data/{benchmark}"
        cc_prefix = f"{str(base_dir)}/cache_and_ckp/{benchmark}_{data_sign}"
        cc_extract_prefix = f"{cc_prefix}/{extract_params.hash()}"
        if fold is not None:
            if fold not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                raise ValueError(f"fold must be in [1, 2, ..., 10], got {fold}")
            cc_extract_prefix += f"-{fold}"
        self.data_sign = data_sign
        self.data_prefix = data_prefix
        self.cc_prefix = cc_prefix
        self.cc_extract_prefix = cc_extract_prefix
        self.extract_params = extract_params
        self._checkpoint_split()

    def _checkpoint_split(self) -> None:
        json_name = "extract_param.json"
        if not Path(f"{self.cc_extract_prefix}/{json_name}").exists():
            JsonHandler.dump_to_file(
                self.extract_params.__dict__,
                f"{self.cc_extract_prefix}/{json_name}",
                indent=2,
            )
            logger.info(f"saved split params to {self.cc_extract_prefix}/{json_name}")
        else:
            logger.info(f"found {self.cc_extract_prefix}/{json_name}")

    def _get_data_sign(self) -> str:
        return get_data_sign(self.benchmark, self.debug)

    def get_data_header(self, q_type: str) -> str:
        return f"{self.data_prefix}/{q_type}_{self.data_sign}.csv"


class TypeAdvisor:
    def __init__(self, q_type: QType):
        self.q_type = q_type

    def get_graph_column(self) -> str:
        if self.q_type in ["q_compile", "q_all"]:
            return "lqp"
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime"]:
            return "qs_lqp"
        if self.q_type in ["qs_pqp_runtime"]:
            return "qs_pqp"
        raise NoQTypeError(self.q_type)

    def get_tabular_columns(self) -> List[str]:
        if self.q_type in ["q_compile"]:
            return ALPHA + THETA_C + THETA_P + THETA_S
        if self.q_type in ["qs_lqp_compile"]:
            return ALPHA_COMPILE + THETA_C + THETA_P + THETA_S
        if self.q_type in ["q_all", "qs_lqp_runtime"]:
            return ALPHA + BETA + GAMMA + THETA_C + THETA_P + THETA_S
        if self.q_type in "qs_pqp_runtime":
            return ALPHA + ALPHA_QS_PLUS + BETA + GAMMA + THETA_C + THETA_S
        raise NoQTypeError(self.q_type)

    def get_decision_columns(self) -> List[str]:
        if self.q_type in ["q_compile", "qs_lqp_compile", "qs_lqp_runtime"]:
            return THETA_C + THETA_P + THETA_S
        if self.q_type in ["q_all"]:
            return THETA_P + THETA_S
        if self.q_type in ["qs_pqp_runtime"]:
            return THETA_S
        raise NoQTypeError(self.q_type)

    def get_tabular_non_decision_columns(self) -> List[str]:
        if self.q_type in ["q_compile"]:
            return ALPHA
        if self.q_type in ["qs_lqp_compile"]:
            return ALPHA_COMPILE
        if self.q_type in ["q_all"]:
            return ALPHA + BETA + GAMMA + THETA_C
        if self.q_type in ["qs_lqp_runtime"]:
            return ALPHA + BETA + GAMMA
        if self.q_type in ["qs_pqp_runtime"]:
            return ALPHA + ALPHA_QS_PLUS + BETA + GAMMA + THETA_C
        raise NoQTypeError(self.q_type)

    def get_objectives(self) -> List[str]:
        if self.q_type in ["q_compile", "q_all"]:
            return ["latency_s", "io_mb"]
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"]:
            return ["latency_s", "io_mb", "ana_latency_s"]
        raise NoQTypeError(self.q_type)

    def get_ag_objectives(self) -> List[str]:
        if self.q_type in ["q_compile", "q_all"]:
            return ["latency_s", "io_mb"]
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"]:
            return ["io_mb", "ana_latency_s"]
        raise NoQTypeError(self.q_type)

    def get_q_type_for_cache(self) -> str:
        if self.q_type in ["q_compile", "q_all"]:
            return self.q_type
        if self.q_type in ["qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"]:
            return "qs"
        raise NoQTypeError(self.q_type)

    def size_mb_in_log(self, operator: Dict) -> float:
        if self.q_type in ["qs_lqp_compile"]:
            x = operator["stats"]["compileTime"]["sizeInBytes"] / 1024.0 / 1024.0
        elif self.q_type in ["qs_lqp_runtime"]:
            x = operator["stats"]["runtime"]["sizeInBytes"] / 1024.0 / 1024.0
        elif self.q_type in ["q_compile", "q_all", "qs_pqp_runtime"]:
            x = operator["sizeInBytes"] / 1024.0 / 1024.0
        else:
            raise NoQTypeError(self.q_type)
        return np.log(np.clip(x, a_min=EPS, a_max=None))

    def rows_count_in_log(self, operator: Dict) -> float:
        if self.q_type in ["qs_lqp_compile"]:
            x = operator["stats"]["compileTime"]["rowCount"] * 1.0
        elif self.q_type in ["qs_lqp_runtime"]:
            x = operator["stats"]["runtime"]["rowCount"] * 1.0
        elif self.q_type in ["q_compile", "q_all", "qs_pqp_runtime"]:
            x = operator["rowCount"] * 1.0
        else:
            raise NoQTypeError(self.q_type)
        return np.log(np.clip(x, a_min=EPS, a_max=None))

    def get_op_name(self, operator: Dict) -> str:
        if self.q_type.startswith("qs_pqp"):
            # drop the suffix "Exec"
            return operator["className"].split(".")[-1][:-4]
        return operator["className"].split(".")[-1]
