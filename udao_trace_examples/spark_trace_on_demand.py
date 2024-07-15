import glob
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import numpy as np

from udao_trace.collector.SparkCollector import SparkCollector
from udao_trace.utils import BenchmarkType, ClusterName, JsonHandler
from udao_trace_examples.params import get_collector_parser


def parse_configuration(j: Dict, fine_grained: bool) -> Dict:
    if "submit_theta" not in j:
        if fine_grained:
            config = {
                **j["theta_c"],
                **j["runtime_theta"]["qs0"]["theta_p"],
                **j["runtime_theta"]["qs0"]["theta_s"],
            }
        else:
            config = {**j["theta_c"], **j["theta_p"], **j["theta_s"]}
    else:
        config = j["submit_theta"]
    return config


def validate_and_parse_configurations(
    spark_collector: SparkCollector,
    config_header: str,
    n_data_per_template: int,
    fine_grained: bool = False,
) -> Tuple[str, List[Dict]]:
    header = spark_collector.header + "/" + config_header
    templates = spark_collector.benchmark.templates
    configurations = []
    for qid in range(1, n_data_per_template + 1):
        for template in templates:
            try:
                j = JsonHandler.load_json(f"{header}/query_{template}-{qid}.json")
                configuration = parse_configuration(j, fine_grained)
            except FileNotFoundError:
                raise ValueError(f"Configuration not found for {template}, {qid}")
            except Exception as e:
                raise e

            for c_name in spark_collector.spark_conf.knob_names:
                if c_name not in configuration:
                    raise ValueError(
                        f"Configuration for {template}, {qid} is missing {c_name}"
                    )
            configurations.append({(template, qid): configuration})
    return header, configurations


def get_mean_std_without_none(
    raw_numbers: List[Optional[float]],
) -> Tuple[float, float]:
    ll = [x for x in raw_numbers if x is not None]
    if len(ll) == 0:
        return 0, 0
    return np.mean(ll).astype(float), np.std(ll).astype(float)


def get_eval_parser() -> ArgumentParser:
    parser = get_collector_parser()
    # fmt: off
    parser.add_argument("--configuration_header", type=str,
                        help="Header for the configurations")
    parser.add_argument("--configuration_name", type=str,
                        help="name for the configurations")
    parser.add_argument("--fine_grained", action="store_true",
                        help="Use fine-grained configurations")
    parser.add_argument("--n_reps", type=int, default=3,
                        help="Number of repetitions for each configuration")
    parser.add_argument("--parse_only", action="store_true",
                        help="Only parse configurations, do not start evaluations")
    parser.add_argument("--force", action="store_true",
                        help="Force parsing even if the file already exists")
    parser.add_argument("--enable_runtime_optimizer", action="store_true",
                        help="Enable runtime optimizer")
    parser.add_argument("--runtime_optimizer_host", type=str, default="localhost",
                        help="Host for runtime optimizer")
    parser.add_argument("--runtime_optimizer_port", type=int, default=12345,
                        help="Port for runtime optimizer")
    # fmt: on

    return parser


if __name__ == "__main__":
    args = get_eval_parser().parse_args()
    if args.trace_header != "evaluations":
        raise ValueError("trace_header must be 'evaluations'")

    spark_collector = SparkCollector(
        knob_meta_file=args.knob_meta_file,
        benchmark_type=BenchmarkType[args.benchmark_type],
        scale_factor=args.scale_factor,
        cluster_name=ClusterName[args.cluster_name],
        parametric_bash_file=args.parametric_bash_file,
        header=args.trace_header,
        debug=args.debug,
        enable_runtime_optimizer=args.enable_runtime_optimizer,
        runtime_optimizer_host=args.runtime_optimizer_host,
        runtime_optimizer_port=args.runtime_optimizer_port,
    )

    header = f"{spark_collector.header}/{args.configuration_header}"
    # ws-/fixed_weights_add_io
    j = JsonHandler.load_json(f"{header}/{args.configuration_name}")
    configurations = []
    for query_id, conf_list in j.items():
        template, qid = query_id.split("-")
        for conf in conf_list:
            conf_dict = {
                k: v
                for k, v in zip(spark_collector.spark_conf.knob_names, conf.split(","))
            }
            pattern = (
                f"{header}/trace_rt_enabled/*_{query_id}_{conf}_*"
                if args.enable_runtime_optimizer
                else f"{header}/trace/*_{query_id}_{conf}_*"
            )
            files = glob.glob(pattern)
            n_exits = len(files)
            for i in range(n_exits, args.n_reps):
                configurations.append({(template, qid): conf_dict})

    print(f"------ Starting evaluations {len(configurations)} configurations")
    spark_collector.start_eval(
        eval_header=header,
        configurations=configurations,
        n_processes=1 if args.enable_runtime_optimizer else args.n_processes,
        cluster_cores=args.cluster_cores,
    )
    print("------ Finished all evaluations")
