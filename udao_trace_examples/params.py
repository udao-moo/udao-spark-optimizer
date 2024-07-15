from argparse import ArgumentParser


def get_base_parser() -> ArgumentParser:
    # fmt: off
    parser = ArgumentParser(description="Spark Trace Collection Script")
    parser.add_argument("--benchmark_type", type=str, default="TPCH",
                        help="Type of benchmark (e.g., TPCH)",)
    parser.add_argument("--scale_factor", type=int, default=100,
                        help="Scale factor of the benchmark")
    parser.add_argument("--n_processes", type=int, default=1,
                        help="number of parallel processors for parsing traces",)
    # fmt: on
    return parser


def get_collector_parser() -> ArgumentParser:
    # fmt: off
    parser = get_base_parser()
    parser.add_argument("--knob_meta_file", type=str,
                        default="assets/spark_configuration_aqe_on.json",
                        help="Path to the knob metadata file")
    parser.add_argument("--cluster_name", type=str, default="HEX1",
                        help="Name of the cluster")
    parser.add_argument("--parametric_bash_file", type=str,
                        default="assets/run_spark_collect.sh",
                        help="Path to the parametric bash file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--default", action="store_true",
                        help="Run default configurations")
    parser.add_argument("--n_data_per_template", type=int, default=10,
                        help="Data points per template, 2273 for TPCH, 490 for TPCDS")
    parser.add_argument("--cluster_cores", type=int, default=150,
                        help="Total available cluster cores")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for randomization")
    parser.add_argument("--trace_header", type=str, default="spark_collector",
                        help="Header for the trace")
    # fmt: on
    return parser


def get_parser_parser() -> ArgumentParser:
    # fmt: off
    parser = get_base_parser()
    parser.add_argument("--header", type=str,
                        default="./spark_collector/tpch100/lhs_22x2273")
    parser.add_argument("--upto", type=int, default=10,
                        help="Name of the traces for each template")
    # fmt: on
    return parser
