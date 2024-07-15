from udao_trace.collector.SparkCollector import SparkCollector
from udao_trace.utils import BenchmarkType, ClusterName
from udao_trace_examples.params import get_collector_parser

if __name__ == "__main__":
    args = get_collector_parser().parse_args()

    spark_collector = SparkCollector(
        knob_meta_file=args.knob_meta_file,
        benchmark_type=BenchmarkType[args.benchmark_type],
        scale_factor=args.scale_factor,
        cluster_name=ClusterName[args.cluster_name],
        parametric_bash_file=args.parametric_bash_file,
        header=args.trace_header,
        debug=args.debug,
    )

    if args.default:
        spark_collector.start_default(
            n_processes=args.n_processes, cluster_cores=args.cluster_cores
        )
    else:
        spark_collector.start_lhs(
            n_data_per_template=args.n_data_per_template,
            n_processes=args.n_processes,
            cluster_cores=args.cluster_cores,
            seed=args.seed,
        )
