from enum import Enum


class VarTypes(Enum):
    INT = "int"
    BOOL = "bool"
    CATEGORY = "category"
    FLOAT = "float"


class ScaleTypes(Enum):
    LOG = "log"
    LINEAR = "linear"


class BenchmarkType(Enum):
    TPCH = "tpch"
    TPCDS = "tpcds"
    TPCXBB = "tpcxbb"


class ClusterName(Enum):
    HEX1 = "hex1"
    HEX2 = "hex2"
    HEX3 = "hex3"
