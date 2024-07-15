from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

from udao_spark.optimizer.atomic_optimizer import AtomicOptimizer
from udao_spark.optimizer.utils import get_ag_meta
from udao_trace.configuration import SparkConf
from udao_trace.utils import BenchmarkType
from udao_trace.workload import Benchmark


def q_compile_setup(base_dir: Path, params: Namespace) -> Dict[str, Any]:
    if params.q_type != "q_compile":
        raise ValueError(f"Diagnosing {params.q_type} is not our focus.")
    if params.hp_choice != "tuned-0215":
        raise ValueError(f"hp_choice {params.hp_choice} is not supported.")
    if params.ensemble and params.graph_choice != "gtn":
        raise ValueError(f"ensembled model only works with {params.graph_choice}.")

    # theta includes 19 decision variables
    decision_variables = (
        ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
        + ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
        + ["s10", "s11"]
    )
    bm, q_type = params.benchmark, params.q_type

    # prepare the traces
    spark_conf = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))
    sample_header = str(base_dir / "assets/query_plan_samples")
    if bm == "tpch":
        benchmark = Benchmark(BenchmarkType.TPCH, params.scale_factor)
        raw_traces = [
            f"{sample_header}/{bm}/{bm}100_{q}-1_1,1g,16,16,48m,200,true,0.6,"
            f"64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,0.2,1024KB"
            f"_application_1701736595646_{2557 + i}.json"
            for i, q in enumerate(benchmark.templates)
        ]
    elif bm == "tpcds":
        benchmark = Benchmark(BenchmarkType.TPCDS, params.scale_factor)
        raw_traces = [
            f"{sample_header}/{bm}/{bm}100_{q}-1_1,1g,16,16,48m,200,true,0.6,"
            f"64MB,0.2,0MB,10MB,200,256MB,5,128MB,4MB,0.2,1024KB"
            f"_application_1701737506122_{3283 + i}.json"
            for i, q in enumerate(benchmark.templates)
        ]
    else:
        raise ValueError(f"benchmark {bm} is not supported")
    for trace in raw_traces:
        print(trace)
        if not Path(trace).exists():
            print(f"{trace} does not exist")
            raise FileNotFoundError(f"{trace} does not exist")

    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    ag_sign = params.ag_sign
    il, bs, tl = params.infer_limit, params.infer_limit_batch_size, params.ag_time_limit
    fold = params.fold
    ag_meta = get_ag_meta(
        bm, hp_choice, graph_choice, q_type, ag_sign, il, bs, tl, fold
    )
    ag_model = {
        "latency_s": params.ag_model_q_latency,
        "io_mb": params.ag_model_q_io,
    }

    # use functions in HierarchicalOptimizer to extract the non-decision inputs
    atomic_optimizer = AtomicOptimizer(
        bm=bm,
        model_sign=ag_meta["model_sign"],
        graph_model_params_path=ag_meta["model_params_path"],
        graph_weights_path=ag_meta["graph_weights_path"],
        q_type=q_type,
        data_processor_path=ag_meta["data_processor_path"],
        spark_conf=spark_conf,
        decision_variables=decision_variables,
        ag_path=ag_meta["ag_path"],
        clf_json_path=None
        if params.disable_failure_clf
        else str(base_dir / f"assets/{bm}_valid_clf_meta.json"),
        clf_recall_xhold=params.clf_recall_xhold,
        verbose=params.verbose,
    )
    return {
        "atomic_optimizer": atomic_optimizer,
        "templates": benchmark.templates,
        "raw_traces": raw_traces,
        "ag_model": ag_model,
    }
