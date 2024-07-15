Prepare Spark programs to support runtime optimization within Spark Adaptive Query Execution (AQE) Framework.

## Spark Recompilation

We re-compiled Apache Spark with a customized version based on a Spark 3.5-SNAPSHOT version, to support (1) trace collection for query plans at multiple granularities, and (2) runtime optimization embedded in adaptive query execution.

Here are the steps to get the customized spark version.

```bash
# using --filter to reduce the size of the log history
export UDAOHOME=/path/to/udao-spark-optimizer
git clone --filter=tree:0 git@github.com:apache/spark.git
cd spark
git checkout 3bc66da6680844cc08cbc181a51d3a8d988b41bb
git am $UDAOHOME/spark-setup/udao.patch

export MAVEN_OPTS="-Xss64m -Xmx2g -XX:ReservedCodeCacheSize=1g"
./dev/make-distribution.sh --name stuning-spark --pip --tgz -Dhadoop.version=3.3.0 -Phive -Phive-thriftserver -Pyarn
# for build locally
# ./build/mvn -DskipTests clean package
```

Get the compiled file at `./spark-3.5.0-SNAPSHOT-bin-stuning-spark.tgz`


## Spark Application Examples for Runtime Optimization

Support TPCH/TPCDS for runtime optimization with our example at https://github.com/Angryrou/spark-stage-tuning.
The compiled jar file is also ready at [spark-stage-tuning_2.12-1.0-SNAPSHOT.jar](./spark-stage-tuning_2.12-1.0-SNAPSHOT.jar)

```bash
git clone git@github.com:Angryrou/spark-stage-tuning.git
cd spark-stage-tuning
sbt package
```

Get the compiled jar file at `target/scala-2.12/spark-stage-tuning_2.12-1.0-SNAPSHOT.jar`
