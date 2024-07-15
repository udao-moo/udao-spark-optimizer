from pathlib import Path

import numpy as np
import pytest

from ...parser.spark_parser import SparkParser
from ...utils import BenchmarkType
from ...utils.logging import _get_logger


@pytest.fixture()
def sp() -> SparkParser:
    return SparkParser(
        benchmark_type=BenchmarkType.TPCH,
        scale_factor=100,
        logger=_get_logger(__name__),
    )


class TestSparkParser:
    @pytest.mark.parametrize(
        "file",
        [
            str(
                Path(__file__).parent.parent
                / "assets"
                / "traces"
                / "tpch100_1-1_1,1g,16,16,48m,200,true,0.6,"
                "64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,"
                "0.2,1024KB_application_1701736595646_2557.json"
            )
        ],
    )
    def test_parse_one_file(self, sp: SparkParser, file: str) -> None:
        q_dict_list, qs_dict_list = sp.parse_one_file(file)
        assert q_dict_list is not None and qs_dict_list is not None
        print()
        print(q_dict_list)
        q_lens = [len(d) for d in q_dict_list]
        assert np.mean(q_lens) == 37 and np.std(q_lens) == 0
        print(qs_dict_list)
        qs_lens = [len(d) for d in qs_dict_list]
        assert np.mean(qs_lens) == 40 and np.std(qs_lens) == 0
