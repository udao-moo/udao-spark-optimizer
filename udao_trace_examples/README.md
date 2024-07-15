Trace Collection Notes
---

## Default Spark Parameters

Most parameters have a default value according to Spark. For the resource parameters that do not have default values, here are our considerations. Our Spark-cluster has 5 worker nodes, each with 30 cores and 700G memory. The resource limit is 80% capacity, i.e., 24 * 5 = 120 cores and 2.8T memory in total. For default
   - in Yarn, `spark.executor.cores=1` by default.
   - `spark.executor.memory=1g` by default.
   - We set `spark.executor.instances=16` in the default setting to support 7 Spark SQLs running in parallel. (120 // (16 + 1) = 7)


## Scripts

Note that a new user will need to adapt the following code their specific Spark cluster setup.

1. trace collector: `spark_trace_collector.py`
   ```bash
   # tpch run
   python spark_trace_collector.py --benchmark_type TPCH --cluster_name HEX1 --n_data_per_template 2273 --n_processes 16 --cluster_cores 120
   # tpcds run
   python spark_trace_collector.py --benchmark_type TPCDS --cluster_name HEX2 --n_data_per_template 490 --n_processes 16 --cluster_cores 120
   ```

2. trace parser: `spark_trace_parser.py`
   ```bash
   # tpch
   python spark_trace_parser.py --header ./spark_collector/tpch100/lhs_22x2273 --benchmark_type TPCH --scale_factor 100 --n_processes 16 --upto 2273
   # tpcds
   python spark_trace_parser.py --header ./spark_collector/tpcds100/lhs_102x490 --benchmark_type TPCDS --scale_factor 100 --n_processes 16 --upto 490
   ```
