import hashlib
from abc import ABC
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, Literal

QType = Literal[
    "q_compile", "q_all", "qs_lqp_compile", "qs_lqp_runtime", "qs_pqp_runtime"
]


@dataclass
class UdaoParams(ABC):
    def hash(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class ExtractParams(UdaoParams):
    lpe_size: int
    vec_size: int
    seed: int
    q_type: QType
    debug: bool = False

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "ExtractParams":
        return cls(**data_dict)

    def hash(self) -> str:
        attributes_tuple = str(
            (
                self.lpe_size,
                self.vec_size,
                self.seed,
            )
        ).encode("utf-8")
        sha256_hash = hashlib.sha256(attributes_tuple)
        hex12 = self.q_type + "/" + sha256_hash.hexdigest()[:12]
        if self.debug:
            return hex12 + "_debug"
        return hex12


def get_base_parser() -> ArgumentParser:
    # fmt: off
    parser = ArgumentParser(description="Udao Script with Input Arguments")
    # Data-related arguments
    parser.add_argument("--benchmark", type=str, default="tpch",
                        help="Benchmark name")
    parser.add_argument("--scale-factor", type=int, default=100)
    parser.add_argument("--q_type", type=str, default="q_compile",
                        choices=["q_compile", "q_all", "qs_lqp_compile",
                                 "qs_lqp_runtime", "qs_pqp_runtime"],
                        help="graph type")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold number, from 0 to 9")
    # fmt: on
    return parser


def _get_graph_base_parser() -> ArgumentParser:
    # fmt: off
    parser = get_base_parser()
    # Common embedding parameters
    parser.add_argument("--lpe_size", type=int, default=8,
                        help="Provided Laplacian Positional encoding size")
    parser.add_argument("--vec_size", type=int, default=16,
                        help="Word2Vec embedding size")
    # Learning parameters
    parser.add_argument("--init_lr", type=float, default=1e-1,
                        help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--loss_weights", nargs="+", type=float, default=None,)
    # Others
    parser.add_argument("--num_workers", type=int, default=15,
                        help="non-debug only")
    parser.add_argument("--op_groups", nargs="+", default=["type", "cbo", "op_enc"],
                        help="List of operation groups")
    # fmt: on
    return parser


def get_graph_avg_params() -> ArgumentParser:
    parser = _get_graph_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser


def get_tree_lstm_params() -> ArgumentParser:
    parser = _get_graph_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--lstm_hidden_dim", type=int, default=32,
                        help="Hidden dimension of the LSTM")
    parser.add_argument("--lstm_dropout", type=float, default=0.0,
                        help="Dropout rate in the LSTM")
    parser.add_argument("--readout", type=str, default="mean",
                        choices=["mean", "max", "sum"],
                        help="Readout function")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser


def get_tree_cnn_params() -> ArgumentParser:
    parser = _get_graph_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--output_size", type=int, default=64,
                        help="Embedder output size")
    parser.add_argument("--tcnn_hidden_dim", type=int, default=256,
                        help="Hidden dimension of the TreeConv")
    parser.add_argument("--readout", type=str, default="max",
                        choices=["mean", "max", "sum"],
                        help="Readout function")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser


def get_qppnet_params() -> ArgumentParser:
    parser = _get_graph_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="Number of layers in Neural Unit of QPPNet")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden dimension of layers in the Neural Unit")
    # fmt: on
    return parser


def get_graph_transformer_params() -> ArgumentParser:
    parser = _get_graph_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--pos_encoding_dim", type=int, default=8,
                        help="Positional encoding dimension for use")
    parser.add_argument("--gtn_n_layers", type=int, default=2,
                        help="Number of layers in the GTN")
    parser.add_argument("--gtn_n_heads", type=int, default=2,
                        help="Number of heads in the GTN")
    parser.add_argument("--readout", type=str, default="mean",
                        choices=["mean", "max", "sum"],
                        help="Readout function")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser


def get_ag_parameters() -> ArgumentParser:
    parser = get_base_parser()
    # fmt: off
    parser.add_argument("--hp_choice", type=str, default="tuned-0215",
                        choices=[
                            "tuned-0215", "tuned-0624",
                            "0624-fold-1", "0624-fold-2", "0624-fold-3", "0624-fold-4",
                            "0624-fold-5", "0624-fold-6", "0624-fold-7", "0624-fold-8",
                            "0624-fold-9", "0624-fold-10"
                        ])
    parser.add_argument("--graph_choice", type=str, default="gtn",
                        choices=["avg", "gtn", "tlstm", "raal", "qppnet", "qf", "none"])
    parser.add_argument("--ag_sign", type=str, default="medium_quality")
    parser.add_argument("--num_gpus", type=int, default=2,)
    parser.add_argument("--infer_limit", type=float, default=None,
                        help="Inference limit, e.g., 1e-5")
    parser.add_argument("--infer_limit_batch_size", type=int, default=None,
                        help="Inference limit batch size, e.g., 50000")
    parser.add_argument("--ag_time_limit", type=int, default=None,
                        help="Time limit in seconds for the AG")
    parser.add_argument("--clf_recall_xhold", type=float, default=0.6,
                        help="The threshold of recall used in the failure classifier. "
                             "Our clf gives Positive signal for failure cases. When it"
                             " gives Negative, we want to reduce a False Negative (FN) "
                             "Therefore, we pick recall=TP/(TP+FN) as the metric. ")
    parser.add_argument("--disable_failure_clf", action="store_true",
                        help="Disable failure classifiers")
    # fmt: on

    return parser


def get_compile_time_optimizer_parameters() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--use_mlp", action="store_true",
                        help="Enable MLP only")
    parser.add_argument("--ag_model_qs_ana_latency", type=str, default=None,
                        help="specific model name for AG for QS_R ana_latency")
    parser.add_argument("--ag_model_qs_io", type=str, default=None,
                        help="specific model name for AG for QS_R IO")

    parser.add_argument("--save_data", action="store_true",
                        help="Enable to save data")
    parser.add_argument("--save_data_header", type=str, default="./output",
                        help="the head of data save path")
    parser.add_argument("--moo_algo", type=str, default="hmooc%B",
                        choices=["hmooc%B", "hmooc%GD",
                                 "hmooc%WS&11",
                                 "evo", "ws", "ppf"],
                        help="Algorithm for the compile-time optimization", )
    parser.add_argument("--n_c_samples", type=int, default=100,
                        help="the number of random samples of theta_c")
    parser.add_argument("--n_p_samples", type=int, default=100,
                        help="the number of random samples of theta_p")

    parser.add_argument("--sample_mode", type=str, default="grid-search",
                        choices=["grid-search", "random", "lhs",
                                 "grid-adaptive-cut"],
                        help="Sample type for HMOOC")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Population size in EVO")
    parser.add_argument("--nfe", type=int, default=1000,
                        help="The number of function evaluations in EVO")
    parser.add_argument("--time_limit", type=int, default=-1,
                        help="Time limit for algorithms in "
                             "the compile-time optimization")

    parser.add_argument("--n_ws", type=int, default=11,
                        help="The number of weight pairs in WS")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="The number of samples in WS"
    )

    parser.add_argument("--n_grids", type=int, default=1,
                        help="The number of grids used in PF-AP")
    parser.add_argument("--n_process", type=int, default=1,
                        help="The number of processes for multiprocessing "
                             "used in PF-AP")
    parser.add_argument("--n_max_iters", type=int, default=1,
                        help="The number of maximum iterations used in PF-AP")

    parser.add_argument("--set_query_control", action="store_true",
                        help="Enable query-level control, "
                             "which is fine-grained control by default")

    # add by chenghao
    parser.add_argument("--conf_save", type=str,
                        default="chenghao_conf_save")
    parser.add_argument("--selected_features", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode")

    # fmt: on
    return parser


def get_runtime_optimizer_parameters() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--use_mlp", action="store_true",
                        help="Enable MLP only")
    parser.add_argument("--ag_model_q_latency", type=str, default=None,
                        help="specific model name for AG for Q_R latency")
    parser.add_argument("--ag_model_q_io", type=str, default=None,
                        help="specific model name for AG for Q_R IO")
    parser.add_argument("--ag_model_qs_ana_latency", type=str, default=None,
                        help="specific model name for AG for QS_R ana_latency")
    parser.add_argument("--ag_model_qs_io", type=str, default=None,
                        help="specific model name for AG for QS_R IO")
    parser.add_argument("--ag_sign_q", type=str, default="medium_quality")
    parser.add_argument("--infer_limit_q", type=float, default=None,
                        help="Inference limit, e.g., 1e-5 (Q)")
    parser.add_argument("--infer_limit_batch_size_q", type=int, default=None,
                        help="Inference limit batch size, e.g., 50000 (Q)")
    parser.add_argument("--ag_time_limit_q", type=int, default=None,
                        help="Time limit in seconds for the AG (Q)")
    parser.add_argument("--ag_sign_qs", type=str, default="medium_quality")
    parser.add_argument("--infer_limit_qs", type=float, default=None,
                        help="Inference limit, e.g., 1e-5 (QS)")
    parser.add_argument("--infer_limit_batch_size_qs", type=int, default=None,
                        help="Inference limit batch size, e.g., 50000 (QS)")
    parser.add_argument("--ag_time_limit_qs", type=int, default=None,
                        help="Time limit in seconds for the AG (QS)")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Enable sanity check")
    parser.add_argument("--sample_mode", type=str, default="random",
                        choices=["random", "grid"],
                        help="Inner solver for the runtime optimization")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--moo_mode", type=str, default="BF",
                        choices=["BF"],
                        help="Moo mode for the runtime optimization")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode")
    parser.add_argument("--selected_features", action="store_true")
    # fmt: on
    return parser
