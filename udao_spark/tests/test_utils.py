import os
from pathlib import Path

from udao.data.handler.data_handler import DataProcessor

from udao_trace.utils import PickleHandler


def test_load_data_handler(ckp_path_qs_lqp_compile: str) -> None:
    data_handler_path = str(
        Path(ckp_path_qs_lqp_compile) / "ea0378f56dcf_debug/data_processor.pkl"
    )
    data_handler = PickleHandler.load(
        os.path.dirname(data_handler_path), os.path.basename(data_handler_path)
    )
    assert isinstance(data_handler, DataProcessor)
