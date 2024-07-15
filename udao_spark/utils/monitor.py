import os
from typing import Dict, List

from udao_trace.utils import JsonHandler


class UdaoMonitor:
    def __init__(self) -> None:
        self.input_extraction_ms: float = 0.0
        self.graph_embedding_computation_ms: float = 0.0

        self.theta_sampling_ms: float = 0.0
        self.theta_numbers: int = 0

        self.model_inference_ms: float = 0.0
        self.pareto_filtering_ms: float = 0.0
        self.utopia_nearest_selection_ms: float = 0.0
        self.output_preparation_ms: float = 0.0

    def to_server(self) -> Dict:
        d = self.to_dict()
        return {
            "server_total_time_ms": d["total_time_ms"],
            "input_extraction_ms": d["breakdown"]["input_extraction_ms"],
            "graph_embedding_computation_ms": d["breakdown"][
                "graph_embedding_computation_ms"
            ],
            "theta_sampling_ms": d["breakdown"]["theta_sampling_ms"],
            "model_inference_ms": d["breakdown"]["model_inference_ms"],
            "pareto_filtering_ms": d["breakdown"]["pareto_filtering_ms"],
            "utopia_nearest_selection_ms": d["breakdown"][
                "utopia_nearest_selection_ms"
            ],
            "output_preparation_ms": d["breakdown"]["output_preparation_ms"],
        }

    def to_dict(self) -> Dict:
        return {
            "total_time_ms": sum(
                [
                    self.input_extraction_ms,
                    self.graph_embedding_computation_ms,
                    self.theta_sampling_ms,
                    self.model_inference_ms,
                    self.pareto_filtering_ms,
                    self.utopia_nearest_selection_ms,
                    self.output_preparation_ms,
                ]
            ),
            "breakdown": {
                "input_extraction_ms": self.input_extraction_ms,
                "graph_embedding_computation_ms": self.graph_embedding_computation_ms,
                "theta_sampling_ms": self.theta_sampling_ms,
                "theta_numbers": self.theta_numbers,
                "model_inference_ms": self.model_inference_ms,
                "pareto_filtering_ms": self.pareto_filtering_ms,
                "utopia_nearest_selection_ms": self.utopia_nearest_selection_ms,
                "output_preparation_ms": self.output_preparation_ms,
            },
        }

    def save_to_file(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        JsonHandler.dump_to_file(self.to_dict(), file_path, indent=2)


class DivAndConqMonitor(UdaoMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.theta_sampling_ms_list: List[float] = []
        self.theta_numbers_list: List[int] = []
        self.model_inference_ms_list: List[float] = []
        self.pareto_filtering_ms_list: List[float] = []
        self.utopia_nearest_selection_ms_list: List[float] = []

    def agg(self) -> None:
        self.theta_sampling_ms = sum(self.theta_sampling_ms_list)
        self.theta_numbers = sum(self.theta_numbers_list)
        self.model_inference_ms = sum(self.model_inference_ms_list)
        self.pareto_filtering_ms = sum(self.pareto_filtering_ms_list)
        self.utopia_nearest_selection_ms = sum(self.utopia_nearest_selection_ms_list)

    def to_dict(self) -> Dict:
        self.agg()
        return {
            "total_time_ms": sum(
                [
                    self.input_extraction_ms,
                    self.graph_embedding_computation_ms,
                    self.theta_sampling_ms,
                    self.model_inference_ms,
                    self.pareto_filtering_ms,
                    self.utopia_nearest_selection_ms,
                ]
            ),
            "breakdown": {
                "input_extraction_ms": self.input_extraction_ms,
                "graph_embedding_computation_ms": self.graph_embedding_computation_ms,
                "theta_sampling_ms": {
                    "total": self.theta_sampling_ms,
                    "steps": self.theta_sampling_ms_list,
                },
                "theta_numbers": {
                    "total": self.theta_numbers,
                    "steps": self.theta_numbers_list,
                },
                "model_inference_ms": {
                    "total": self.model_inference_ms,
                    "steps": self.model_inference_ms_list,
                },
                "pareto_filtering_ms": {
                    "total": self.pareto_filtering_ms,
                    "steps": self.pareto_filtering_ms_list,
                },
                "utopia_nearest_selection_ms": {
                    "total": self.utopia_nearest_selection_ms,
                    "steps": self.utopia_nearest_selection_ms_list,
                },
                "output_preparation_ms": self.output_preparation_ms,
            },
        }
