from pathlib import Path

import pytest

from ..model.model_server import ModelServer


@pytest.fixture
def ms_graph_avg(ckp_path_qs_lqp_compile: Path) -> ModelServer:
    ckp_path = ckp_path_qs_lqp_compile / "ea0378f56dcf_debug/graph_avg_082b46304679"
    model_params_path = str(ckp_path / "model_struct_params.json")
    weights_path = str(
        ckp_path
        / "learning_74b9b64f5030"
        / "1-val_latency_s_WMAPE=0.934-val_io_mb_WMAPE=1.000"
        "-val_ana_latency_s_WMAPE=0.987.ckpt"
    )
    return ModelServer.from_ckp_path(
        model_sign="graph_avg",
        model_params_path=model_params_path,
        weights_path=weights_path,
    )


class TestModelServer:
    def test_graph_avg_from_ckp_path(self, ms_graph_avg: ModelServer) -> None:
        assert ms_graph_avg.model is not None
        assert ms_graph_avg.objectives == ["latency_s", "io_mb", "ana_latency_s"]
        assert ms_graph_avg.model_sign == "graph_avg"
