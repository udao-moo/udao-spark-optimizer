"""
To build to DataFrame from the raw traces in one path

DF1 (q.csv)
appid, tid, qid, lqp_id, lqp(json), im-1, im-2, pd(a list object),
ss-1, ss-2, ..., ss-8, k1,..., k8, s1,..., s9, latency_s, io_mb

DF2 (qs.csv)
appid, tid, qid, qs_id, qs_lqp(json), qs_pqp(json), initial_part_size,
im-1, im-2, pd(a list object), ss-1, ss-2, ..., ss-8, k1,..., k8, s1,..., s9,
latency_s, io_mb, ana_latency_s

"""

from udao_trace.parser.spark_parser import SparkParser
from udao_trace.utils import BenchmarkType
from udao_trace.utils.logging import _get_logger
from udao_trace_examples.params import get_parser_parser

if __name__ == "__main__":
    args = get_parser_parser().parse_args()

    logger = _get_logger(__name__)
    sp = SparkParser(
        benchmark_type=BenchmarkType[args.benchmark_type],
        scale_factor=args.scale_factor,
        logger=logger,
    )
    templates = sp.benchmark.templates
    df_q, df_qs = sp.parse(
        header=args.header,
        templates=templates,
        upto=args.upto,
        n_processes=args.n_processes,
    )
