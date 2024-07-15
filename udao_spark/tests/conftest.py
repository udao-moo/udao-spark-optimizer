from pathlib import Path

import pytest

from udao_trace.configuration import SparkConf


@pytest.fixture
def ckp_path_qs_lqp_compile() -> Path:
    base_dir = Path(__file__).parent
    return (
        base_dir.parent.parent
        / "playground"
        / "cache_and_ckp/tpch_22x10/qs_lqp_compile"
    )


@pytest.fixture
def sc() -> SparkConf:
    return SparkConf(
        str(Path(__file__).parent / "assets/spark_configuration_aqe_on.json")
    )
