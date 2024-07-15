#!/bin/bash

# Initialize variables with default values
query="9 1"
conf="1 1g 16 16 48m 200 true 0.6 64MB 0.2 0MB 10MB 200 256MB 5.0 128MB 4MB 0.2 1024KB"
name="unnamed"
runtime="false localhost 12345"
xpath="./outs"
verbose_mode=false

# Define usage function
usage() {
  echo "Usage: $0 -q <query> -c <conf> -b <benchmark> -n <name> -r <runtime> -x <xpath> [-v]"
  echo "  -q <query>: Specify template id (tid) and query variant id (qid)."
  echo "  -c <conf>: Specify context parameters from k1 to k8 + s1 to s11"
  echo "  -b <benchmark>: Specify the benchmark of running TPC-query, e.g., tpch, tpcds"
  echo "  -n <name>: Specify the name of spark app"
  echo "  -r <runtime>: Specify the runtime optimizer"
  echo "  -x <xpath>: Specify the path for the extracted traces"
  echo "  -v: Enable verbose mode."
  exit 1
}

# Parse command line options with getopts
while getopts "q:c:b:n:r:x:v" opt; do
  case "$opt" in
    q) query="$OPTARG";;
    c) conf="$OPTARG";;
    b) benchmark="$OPTARG";;
    n) name="$OPTARG";;
    r) runtime="$OPTARG";;
    x) xpath="$OPTARG";;
    v) verbose_mode=true;;
    \?) usage;;
  esac
done
read tid qid <<< "$query"
read k1 k2 k3 k4 k5 k6 k7 k8 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 <<< "$conf"
read if_runtime runtime_host runtime_port <<< "$runtime"

spath=/opt/hex_users/$USER/chenghao/spark-stage-tuning
jpath=/opt/hex_users/$USER/spark-3.2.1-hadoop3.3.0/jdk1.8
lpath=/opt/hex_users/$USER/chenghao/spark-stage-tuning/src/main/resources/log4j2.properties
qpath=/opt/hex_users/$USER/chenghao/UDAO2022

~/spark/bin/spark-submit \
--class edu.polytechnique.cedar.spark.benchmark.RunTemplateQueryForRuntime \
--name ${name} \
--master yarn \
--deploy-mode client \
--conf spark.executorEnv.JAVA_HOME=${jpath} \
--conf spark.yarn.appMasterEnv.JAVA_HOME=${jpath} \
--conf spark.executor.cores=${k1} \
--conf spark.executor.memory=${k2} \
--conf spark.executor.instances=${k3} \
--conf spark.default.parallelism=${k4} \
--conf spark.reducer.maxSizeInFlight=${k5} \
--conf spark.shuffle.sort.bypassMergeThreshold=${k6} \
--conf spark.shuffle.compress=${k7} \
--conf spark.memory.fraction=${k8} \
--conf spark.yarn.am.cores=${k1} \
--conf spark.yarn.am.memory=${k2} \
--conf spark.sql.adaptive.enabled=true \
--conf spark.driver.maxResultSize=0 \
--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=${s1} \
--conf spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin=${s2} \
--conf spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold=${s3} \
--conf spark.sql.autoBroadcastJoinThreshold=${s4} \
--conf spark.sql.adaptive.autoBroadcastJoinThreshold=${s4} \
--conf spark.sql.shuffle.partitions=${s5} \
--conf spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes=${s6} \
--conf spark.sql.adaptive.skewJoin.skewedPartitionFactor=${s7} \
--conf spark.sql.files.maxPartitionBytes=${s8} \
--conf spark.sql.files.openCostInBytes=${s9} \
--conf spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor=${s10} \
--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=${s11} \
--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false \
--conf spark.sql.parquet.compression.codec=snappy \
--conf spark.sql.broadcastTimeout=10000 \
--conf spark.rpc.askTimeout=12000 \
--conf spark.shuffle.io.retryWait=60 \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.kryoserializer.buffer.max=512m \
--driver-java-options "-Dlog4j.configuration=file:$lpath" \
--conf "spark.driver.extraJavaOptions=-Xms20g" \
--conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \
--files "$lpath" \
--jars ~/spark/examples/jars/scopt_2.12-3.7.1.jar \
$spath/target/scala-2.12/spark-stage-tuning_2.12-1.0-SNAPSHOT.jar \
-b $benchmark -t ${tid} -q ${qid} -s 100 -x ${xpath} \
-u ${if_runtime} --runtimeSolverHost ${runtime_host} --runtimeSolverPort ${runtime_port} \
-l ${qpath}/resources/${benchmark}-kit/spark-sqls -v $verbose_mode
