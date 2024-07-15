import glob
import os
from itertools import chain
from logging import Logger
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils import BenchmarkType, JsonHandler
from ..utils.logging import _get_logger
from ..workload import Benchmark

THETA_C = [
    "spark.executor.cores",
    "spark.executor.memory",
    "spark.executor.instances",
    "spark.default.parallelism",
    "spark.reducer.maxSizeInFlight",
    "spark.shuffle.sort.bypassMergeThreshold",
    "spark.shuffle.compress",
    "spark.memory.fraction",
]
THETA_P = [
    "spark.sql.adaptive.advisoryPartitionSizeInBytes",
    "spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin",
    "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold",
    "spark.sql.adaptive.autoBroadcastJoinThreshold",
    "spark.sql.shuffle.partitions",
    "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor",
    "spark.sql.files.maxPartitionBytes",
    "spark.sql.files.openCostInBytes",
]
THETA_S = [
    "spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor",
    "spark.sql.adaptive.coalescePartitions.minPartitionSize",
]


def parse_conf(conf: Dict) -> Dict:
    d = {
        f"{k}": v
        for k_type, k_list in conf.items()
        for kv in k_list
        for k, v in kv.items()
    }
    return {
        **{f"theta_c-{k}": d[k] for k in THETA_C if k in d},
        **{f"theta_p-{k}": d[k] for k in THETA_P if k in d},
        **{f"theta_s-{k}": d[k] for k in THETA_S if k in d},
    }


def parse_base(d: Dict) -> Tuple[Dict, Dict]:
    im = {f"IM-{k}": v for k, v in d["IM"].items()}
    pd = d["PD"]
    ss_dict = {}
    if "RunningQueryStageSnapshot" in d:
        for k, v in d["RunningQueryStageSnapshot"].items():
            if isinstance(v, list):
                for tile, v_ in zip([0, 25, 50, 75, 100], v):
                    ss_dict[f"SS-{k}-{tile}tile"] = v_
            else:
                ss_dict[f"SS-{k}"] = v
    else:
        ss_dict = {
            "SS-RunningTasksNum": 0,
            "SS-FinishedTasksNum": 0,
            "SS-FinishedTasksTotalTimeInMs": 0.0,
            "SS-FinishedTasksDistributionInMs-0tile": 0.0,
            "SS-FinishedTasksDistributionInMs-25tile": 0.0,
            "SS-FinishedTasksDistributionInMs-50tile": 0.0,
            "SS-FinishedTasksDistributionInMs-75tile": 0.0,
            "SS-FinishedTasksDistributionInMs-100tile": 0.0,
        }
    if "Configuration" in d:
        conf_key = "Configuration"
    else:
        assert "RuntimeConfiguration" in d
        conf_key = "RuntimeConfiguration"
    conf = parse_conf(d[conf_key])
    return {**im, **{"PD": pd}, **ss_dict}, conf


def parse_lqp_objectives(d: Dict) -> Dict:
    return {
        "latency_s": d["DurationInMs"] / 1000,
        "io_mb": d["IOBytes"]["Total"] / 1024 / 1024,
    }


def parse_qs_objectives(d: Dict) -> Dict:
    return {
        "latency_s": d["DurationInMs"] / 1000,
        "io_mb": d["IOBytes"]["Total"] / 1024 / 1024,
        "total_task_duration_s": d["TotalTasksDurationInMs"] / 1000,
    }


def drop_raw_plan(d: Dict, to_drop: str = "rawPlan") -> Dict:
    return {k: v for k, v in d.items() if k != to_drop}


def parse_lqp_features(d: Dict) -> Tuple[Dict, Dict]:
    lqp_str = JsonHandler.dump_to_string(drop_raw_plan(d["LQP"]), indent=None)
    base, conf = parse_base(d)
    return {**{"lqp": lqp_str}, **base}, conf


def parse_lqp(
    feat: Dict, obj: Dict, meta: Dict, theta_c: Optional[Dict] = None
) -> Dict:
    feat_dict, conf = parse_lqp_features(feat)
    obj_dict = parse_lqp_objectives(obj)
    if theta_c is None:
        return {**meta, **feat_dict, **conf, **obj_dict}
    else:
        return {**meta, **feat_dict, **theta_c, **conf, **obj_dict}


def parse_qs(d: Dict, meta: Dict, theta_c: Dict) -> Dict:
    qs_lqp_str = JsonHandler.dump_to_string(drop_raw_plan(d["QSLogical"]), indent=None)
    qs_pqp_str = JsonHandler.dump_to_string(drop_raw_plan(d["QSPhysical"]), indent=None)
    local = {
        "qs_lqp": qs_lqp_str,
        "qs_pqp": qs_pqp_str,
        "InitialPartitionNum": d["InitialPartitionNum"],
    }
    base, conf = parse_base(d)
    obj_dict = parse_qs_objectives(d["Objectives"])
    return {**meta, **local, **base, **theta_c, **conf, **obj_dict}


class SparkParser:
    def __init__(
        self,
        benchmark_type: BenchmarkType,
        scale_factor: int,
        logger: Logger,
    ):
        benchmark = Benchmark(benchmark_type=benchmark_type, scale_factor=scale_factor)
        self.benchmark = benchmark
        self.benchmark_prefix = benchmark.get_prefix()
        self.logger = logger

    def parse_one_file(self, file: str) -> Tuple[Optional[List], Optional[List]]:
        try:
            appid = "application_" + file.split("_application_")[-1][:-5]
            tq = file.split("/")[-1].split("_")[1]
            template, qid = tq.split("-")
            meta = {"appid": appid, "template": template, "qid": qid}

            d = JsonHandler.load_json(file)
            # work on compile_time LQP
            q_dict_list = []
            compile_time_lqp_dict = parse_lqp(
                d["CompileTimeLQP"], d["Objectives"], {**meta, **{"lqp_id": 0}}, None
            )
            theta_c = {
                k: v
                for k, v in compile_time_lqp_dict.items()
                if k.startswith("theta_c")
            }
            q_dict_list.append(compile_time_lqp_dict)
            # work on runtime LQP
            for lqp_id, runtime_lqp in d["RuntimeLQPs"].items():
                runtime_lqp_dict = parse_lqp(
                    runtime_lqp,
                    runtime_lqp["Objectives"],
                    {**meta, **{"lqp_id": lqp_id}},
                    theta_c,
                )
                q_dict_list.append(runtime_lqp_dict)

            # work on QSs
            qs_dict_list = []
            for qs_id, qs in d["RuntimeQSs"].items():
                qs_dict = parse_qs(qs, {**meta, **{"qs_id": qs_id}}, theta_c)
                qs_dict = {**qs_dict, **theta_c}
                qs_dict_list.append(qs_dict)

            return q_dict_list, qs_dict_list
        except Exception as e:
            self.logger.error(f"failed to parse {file} with error: {e}")
            return None, None

    @staticmethod
    def prepare_headers(header: str) -> Tuple[str, str]:
        trace_header = f"{header}/trace"
        assert os.path.exists(trace_header), FileNotFoundError(trace_header)
        csv_header = f"{header}/csv"
        os.makedirs(csv_header, exist_ok=True)
        return trace_header, csv_header

    def parse(
        self, header: str, templates: List[str], upto: int = 10, n_processes: int = 12
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        trace_header, csv_header = self.prepare_headers(header)
        logger = self.logger
        tq2path: Dict[str, Optional[str]] = {
            f"{template}-{qid}": None
            for template in templates
            for qid in range(1, upto + 1)
        }
        for i, tq in enumerate(tq2path):
            trace_prefix = f"{trace_header}/{self.benchmark_prefix}_{tq}_"
            files = glob.glob(f"{trace_prefix}*.json")
            if len(files) == 0:
                logger.warning(f"{trace_prefix}* does not exist")
                continue
            elif len(files) > 1:
                raise Exception(f"multiple files found for {trace_prefix}*")
            else:
                tq2path[tq] = files[0]

        valid_paths = [path for tq, path in tq2path.items() if path is not None]
        tq_parsed = len(valid_paths)
        tq_not_found = len(tq2path) - tq_parsed
        tp_total = len(tq2path)
        logger.info(
            f"queries parsed|not_found|total: {tq_parsed}|{tq_not_found}|{tp_total}"
        )

        logger.info(f"start generating dfs with {n_processes} processes")
        if n_processes > 1:
            with Pool(n_processes) as p:
                results = p.map(self.parse_one_file, valid_paths)
        else:
            results = [self.parse_one_file(path) for path in valid_paths]

        logger.info("start flat join the results...")
        failed = len([item for item in results if item[0] is None])
        logger.info(
            f"parsed queries succeed|failed|total: "
            f"{tq_parsed - failed}|{failed}|{tq_parsed}"
        )
        q_dict_list = list(
            chain.from_iterable(item[0] for item in results if item[0] is not None)
        )
        qs_dict_list = list(
            chain.from_iterable(item[1] for item in results if item[1] is not None)
        )
        df_q = pd.DataFrame(q_dict_list)
        df_qs = pd.DataFrame(qs_dict_list)
        logger.info(f"get df_q {df_q.shape}")
        logger.info(f"get df_qs {df_qs.shape}")

        df_q.to_csv(
            f"{csv_header}/q_{len(templates)}x{upto}.csv", index=False, header=True
        )
        df_qs.to_csv(
            f"{csv_header}/qs_{len(templates)}x{upto}.csv", index=False, header=True
        )
        return df_q, df_qs


if __name__ == "__main__":
    sp = SparkParser(
        benchmark_type=BenchmarkType.TPCH,
        scale_factor=100,
        logger=_get_logger(__name__),
    )
    templates = sp.benchmark.templates
    sp.parse(header="spark_collector/tpch100/lhs_22x2273", templates=templates, upto=10)
