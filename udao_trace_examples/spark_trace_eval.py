import glob
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from udao_trace.collector.SparkCollector import SparkCollector
from udao_trace.parser.spark_parser import parse_lqp_objectives
from udao_trace.utils import BenchmarkType, ClusterName, JsonHandler
from udao_trace_examples.params import get_collector_parser

s3 = "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold"
s4 = "spark.sql.autoBroadcastJoinThreshold"
s5 = "spark.sql.shuffle.partitions"

UdaoNumeric = Union[float, pd.Series, np.ndarray]

aws_cost_cpu_hour_ratio = 0.052624
aws_cost_mem_hour_ratio = 0.0057785  # for GB*H
io_gb_ratio = 0.02  # for GB


def get_cloud_cost_wo_io(
    lat: UdaoNumeric, cores: UdaoNumeric, mem: UdaoNumeric, nexec: UdaoNumeric
) -> UdaoNumeric:
    cpu_hour = (nexec + 1) * cores * lat / 3600.0
    mem_hour = (nexec + 1) * mem * lat / 3600.0
    cost = cpu_hour * aws_cost_cpu_hour_ratio + mem_hour * aws_cost_mem_hour_ratio
    return cost


def get_cloud_cost_add_io(
    cost_wo_io: UdaoNumeric,
    io_mb: UdaoNumeric,
) -> UdaoNumeric:
    return cost_wo_io + io_mb / 1024.0 * io_gb_ratio


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

    if args.fine_grained:
        raise NotImplementedError(
            "theta_p should be extracted based on QS with join operators"
        )

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

    if args.default:
        header = f"{spark_collector.header}/default_conf"
        configuration = {
            k.name: k.default for k in spark_collector.spark_conf.knob_list
        }
        configurations = []
        for template in spark_collector.benchmark.templates:
            for qid in range(1, args.n_data_per_template + 1):
                configurations.append({(template, qid): configuration})
    else:
        header, configurations = validate_and_parse_configurations(
            spark_collector=spark_collector,
            config_header=args.configuration_header,
            n_data_per_template=args.n_data_per_template,
            fine_grained=args.fine_grained,
        )

    if not args.parse_only:
        print("------ Starting evaluations")
        for i in range(args.n_reps):
            print("------ Starting evaluation", i + 1)
            spark_collector.start_eval(
                eval_header=header,
                configurations=configurations,
                n_processes=1 if args.enable_runtime_optimizer else args.n_processes,
                cluster_cores=args.cluster_cores,
            )
        print("------ Finished all evaluations")

    print("------ Start parsing")
    if args.enable_runtime_optimizer:
        file_name = f"{header}/evaluations_with_runtime_optimizer.json"
    else:
        file_name = f"{header}/evaluations.json"

    if args.force:
        print("------ Force parsing")
    else:
        if os.path.exists(file_name):
            print("------ File already exists")
            try:
                j = JsonHandler.load_json(file_name)
                print(j["agg_stats"])
                exit(0)
            except Exception as e:
                print(f"------ File is corrupted {e}, force parsing")

    stats_dict = {}
    agg_stats_dict = {}
    for conf in configurations:
        (template, qid), configuration = list(conf.items())[0]
        if args.enable_runtime_optimizer:
            json_files = glob.glob(
                f"{header}/trace_rt_enabled/*_{template}-{qid}_*.json"
            )
        else:
            json_files = glob.glob(f"{header}/trace/*_{template}-{qid}_*.json")
        print(f"------ Parsing {template}-{qid}, found {len(json_files)} files")
        stats: Dict[str, List[Any]] = defaultdict(list)
        for json_file in json_files:
            try:
                d = JsonHandler.load_json(json_file)
            except Exception as e:
                print(f"Skip due to - failed to load {json_file} with error: {e}")
                continue
                # raise Exception(f"Failed to load {json_file} with error: {e}")
            obj_dict = parse_lqp_objectives(d["Objectives"])
            lat_s = obj_dict["latency_s"]
            io_mb = obj_dict["io_mb"]
            total_cores = int(configuration["spark.executor.cores"]) * (
                int(configuration["spark.executor.instances"]) + 1
            )
            cost_wo_io = get_cloud_cost_wo_io(
                lat=lat_s,
                cores=int(configuration["spark.executor.cores"]),
                mem=int(configuration["spark.executor.memory"].replace("g", "")),
                nexec=int(configuration["spark.executor.instances"]),
            )
            cost_w_io = get_cloud_cost_add_io(
                cost_wo_io=cost_wo_io,
                io_mb=io_mb,
            )
            stats["json_trace"].append(json_file)
            stats["latency_s"].append(lat_s)
            stats["io_mb"].append(io_mb)
            stats["cores"].append(total_cores)
            stats["cost_wo_io"].append(cost_wo_io)
            stats["cost_w_io"].append(cost_w_io)
        stats_dict[(template, qid)] = stats
        agg_stats_dict[(template, qid)] = {
            "latency_s": get_mean_std_without_none(stats["latency_s"]),
            "io_mb": get_mean_std_without_none(stats["io_mb"]),
            "cores": get_mean_std_without_none(stats["cores"]),
            "cost_wo_io": get_mean_std_without_none(stats["cost_wo_io"]),
            "cost_w_io": get_mean_std_without_none(stats["cost_w_io"]),
        }
    JsonHandler.dump_to_file(
        {
            "stats": {f"{k[0]}-{k[1]}": v for k, v in stats_dict.items()},
            "agg_stats": {f"{k[0]}-{k[1]}": v for k, v in agg_stats_dict.items()},
        },
        file_name,
        indent=2,
    )
    j = JsonHandler.load_json(file_name)
    print(j["agg_stats"])
