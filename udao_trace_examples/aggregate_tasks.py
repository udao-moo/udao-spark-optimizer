from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, List

import pandas as pd

from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType, JsonHandler, PickleHandler
from udao_trace.workload import Benchmark
from udao_trace_examples.spark_trace_on_demand import parse_configuration

s3 = "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold"
s4 = "spark.sql.autoBroadcastJoinThreshold"
s5 = "spark.sql.shuffle.partitions"


def get_agg_task_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="tpch")
    parser.add_argument("--setting", type=str, default="divB_new_grids")
    parser.add_argument("--prefs", nargs="+", type=int, default=[0, 1, 5, 9, 10])
    return parser


def extract_configure_from_fine_grained(file_path: str, join_ids: List[str]) -> str:
    j = JsonHandler.load_json(file_path)
    theta_c = j["theta_c"]
    theta_p = j["runtime_theta"]["qs0"]["theta_p"]
    theta_s = j["runtime_theta"]["qs0"]["theta_s"]

    if len(join_ids) > 0:
        theta_p[s3] = "{}MB".format(
            min(
                int(j["runtime_theta"][f"qs{ji}"]["theta_p"][s3][:-2])
                for ji in join_ids
            )
        )
        theta_p[s4] = "{}MB".format(
            max(
                10,
                min(
                    int(j["runtime_theta"][f"qs{ji}"]["theta_p"][s4][:-2])
                    for ji in join_ids
                ),
            )
        )
        theta_p[s5] = str(
            max(int(j["runtime_theta"][f"qs{ji}"]["theta_p"][s5]) for ji in join_ids)
        )
        # theta_p[s3] = "{}MB".format(min(
        #     int(k["theta_p"][s3][:-2]) for k in j["runtime_theta"].values()
        # ))
        # theta_p[s4] = "{}MB".format(min(
        #     int(k["theta_p"][s4][:-2]) for k in j["runtime_theta"].values()
        # ))
        # theta_p[s5] = str(max(int(k["theta_p"][s5])
        #                       for k in j["runtime_theta"].values()))
    theta = {**theta_c, **theta_p, **theta_s}
    return ",".join(theta[c] for c in spark_conf.knob_names)


def get_sub_df(df: pd.DataFrame, col_name: str, col_value: Any) -> pd.DataFrame:
    return df[df[col_name] == col_value]


params = get_agg_task_parser().parse_args()
bm = params.benchmark
setting = params.setting
benchmark = Benchmark(BenchmarkType(bm))
base_dir = Path(__file__).parent
spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
decision_variables = (
    ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
    + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
    + ["s10", "s11"]
)
# Prepare traces
sample_header = str(base_dir / "assets/query_plan_samples")
if bm == "tpch":
    raw_traces = [
        f"{sample_header}/{bm}/{bm}100_{q}-1_1,1g,16,16,48m,200,true,0.6,"
        f"64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,0.2,1024KB"
        f"_application_1701736595646_{2557 + i}.json"
        for i, q in enumerate(benchmark.templates)
    ]
elif bm == "tpcds":
    raw_traces = [
        f"{sample_header}/{bm}/{bm}100_{q}-1_1,1g,16,16,48m,200,true,0.6,"
        f"64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,0.2,1024KB"
        f"_application_1701737506122_{3283 + i}.json"
        for i, q in enumerate(benchmark.templates)
    ]
else:
    raise ValueError(f"benchmark {bm} is not supported")


# step 1: target the subQs using join for each query
query_to_join_subqs = defaultdict(list)
for template, raw_trace in zip(benchmark.templates, raw_traces):
    j = JsonHandler.load_json(raw_trace)
    join_ids = []
    for qs_id, v in j["RuntimeQSs"].items():
        if ".Join" in JsonHandler.dump_to_string(v["QSLogical"]):
            join_ids.append(qs_id)
    query_to_join_subqs[template] = join_ids

# step 2: get all recommended configurations in a big DataFrame
pref_dict = {}
prefs = params.prefs
for lat_pref in prefs:
    header = (
        f"evaluations/{bm}100/{params.setting}/{lat_pref/10:.1f}_{(10-lat_pref)/10:.1f}"
    )
    data = []
    for template in benchmark.templates:
        query_id = f"{template}-1"
        if setting == "ws-":
            j = JsonHandler.load_json(f"{header}/query_{query_id}.json")
            theta = parse_configuration(j, False)
            conf = ",".join(theta[c] for c in spark_conf.knob_names)
        else:
            conf = extract_configure_from_fine_grained(
                f"{header}/query_{query_id}.json",
                join_ids=query_to_join_subqs[template],
            )
        data.append([query_id, conf])
    pref_dict[lat_pref] = pd.DataFrame(data, columns=["query_id", "conf"])

df = pd.concat(pref_dict, axis=0)
print(df.shape, df.drop_duplicates().shape)

torun_dict = defaultdict(list)
for i, row in df.drop_duplicates().iterrows():
    query_id, conf = row["query_id"], row["conf"]
    torun_dict[query_id].append(conf)

suffix = "_".join(map(str, prefs)) + "_s4_lb_10MB"
JsonHandler.dump_to_file(
    torun_dict,
    f"evaluations/{bm}100/{setting}/on_demand/distinct_{suffix}.json",
    indent=2,
)
PickleHandler.save(
    pref_dict,
    f"evaluations/{bm}100/{setting}/",
    f"pref_to_df_{suffix}.pkl",
    overwrite=True,
)
