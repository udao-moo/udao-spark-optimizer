import os
import random
import subprocess
import sys
import time
from multiprocessing import Manager, Pool
from typing import Callable, Dict, List, Tuple

import pandas as pd

from ..configuration import SparkConf
from ..environment.cluster import Cluster
from ..utils import BenchmarkType, ClusterName, PickleHandler
from ..utils.handler import error_handler
from ..utils.logging import logger
from ..workload import Benchmark, QueryMatrix


class SparkCollector:
    def __init__(
        self,
        knob_meta_file: str,
        benchmark_type: BenchmarkType,
        scale_factor: int,
        cluster_name: ClusterName,
        parametric_bash_file: str = "assets/run_spark_tpc.sh",
        header: str = "spark_collector",
        debug: bool = False,
        enable_runtime_optimizer: bool = False,
        runtime_optimizer_host: str = "localhost",
        runtime_optimizer_port: int = 12345,
    ):
        spark_conf = SparkConf(knob_meta_file)
        benchmark = Benchmark(benchmark_type=benchmark_type, scale_factor=scale_factor)
        cluster = Cluster(cluster_name)

        self.spark_conf = spark_conf
        self.benchmark = benchmark
        self.cluster = cluster
        self.header = f"{header}/{benchmark.get_prefix()}"
        os.makedirs(self.header, exist_ok=True)

        assert os.path.exists(parametric_bash_file), FileNotFoundError(
            parametric_bash_file
        )
        self.parametric_bash_file = parametric_bash_file

        self.debug = debug
        self.enable_runtime_optimizer = "true" if enable_runtime_optimizer else "false"
        self.runtime_optimizer_host = runtime_optimizer_host
        self.runtime_optimizer_port = runtime_optimizer_port

    @staticmethod
    def _exec_cmd(template: str, qid: int, cmd: str) -> bool:
        logger.debug(f"[{template}-{qid}]: {cmd}")
        try:
            # Run the command and capture both stdout and stderr
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
            )

            # Check the return code
            if result.returncode == 0:
                logger.debug(f"[{template}-{qid}]: success")
                return True
            else:
                logger.error(
                    f"Error executing command. Return code: {result.returncode}"
                )
                logger.error("Error message:")
                print(result.stderr)
                logger.error(
                    f"[{template}-{qid}]: failed, return code: {result.returncode}, "
                    f"error message: {result.stderr}"
                )
                return False
        except Exception as e:
            print(f"Exception: {e}")
            sys.exit(1)

    def _submit(
        self, template: str, qid: int, knob_sign: str, cores: int, header: str
    ) -> None:
        rt_enable, rt_host, rt_port = (
            self.enable_runtime_optimizer,
            self.runtime_optimizer_host,
            self.runtime_optimizer_port,
        )

        trace_header = f"{header}/trace"
        log_header = f"{header}/log"
        app_name = self.benchmark.get_prefix() + f"_{template}-{qid}_{knob_sign}"

        if rt_enable == "true":
            trace_header += "_rt_enabled"
            log_header += "_rt_enabled"
            app_name += "_rt_enabled"
        os.makedirs(trace_header, exist_ok=True)
        os.makedirs(log_header, exist_ok=True)

        logger.info(f"-[{template}-{qid}]: start running {app_name}")
        start = time.time()
        conf_str = knob_sign.replace(",", " ")

        exec_str = (
            f'bash {self.parametric_bash_file} -q "{template} {qid}" -c "{conf_str}" '
            f"-b {self.benchmark.get_name()} -n {app_name} "
            f'-r "{rt_enable} {rt_host} {rt_port}" '
            f"-x {trace_header} 2>&1 "
            f"| tee {log_header}/{template}-{qid}.log"
        )

        logger.debug(f"[{template}-{qid}]: {exec_str}")
        if self.debug:
            time.sleep(random.randint(1, 5))
            success = True
        else:
            success = self._exec_cmd(template, qid, exec_str)
        dt = time.time() - start
        logger.debug(f"[{template}-{qid}]: finished, took {dt:0f}s")

        with self.lock:
            self.current_cores.value -= cores
            if not success:
                self.shared_failure_list.append((template, qid, knob_sign))
            logger.info(
                f"-[{template}-{qid}]: finished, took {dt:0f}s, "
                f"current_cores={self.current_cores.value}"
            )

    def _start(
        self,
        total: int,
        header: str,
        get_next: Callable[[int], Tuple[str, int, str, int]],
        cluster_cores: int = 120,
        n_processes: int = 6,
    ) -> None:
        m = Manager()
        self.lock = m.RLock()
        self.current_cores = m.Value("i", 0)
        self.shared_failure_list = m.list()

        submit_index = 0
        pool = Pool(processes=n_processes)
        while submit_index < total:
            template, qid, knob_sign, cores = get_next(submit_index)
            with self.lock:
                if cores + self.current_cores.value < cluster_cores:
                    self.current_cores.value += cores
                    if_submit = True
                    logger.info(
                        f"Main Process: submit {template}-{qid} for {knob_sign}, "
                        f"current_cores = {self.current_cores.value}"
                    )
                else:
                    if_submit = False
            if if_submit:
                pool.apply_async(
                    func=self._submit,
                    args=(template, qid, knob_sign, cores, header),
                    error_callback=error_handler,
                )
                submit_index += 1
            time.sleep(1)
        pool.close()
        pool.join()

        logger.info(
            f"Total {total} queries submitted, {len(self.shared_failure_list)} failed:"
        )
        logger.info(self.shared_failure_list)

    def _get_lhs_conf_dict(self, n_data_per_template: int) -> Dict[str, pd.DataFrame]:
        cache_header = f"{self.header}/cache/template_to_conf_dict"
        spark_conf = self.spark_conf
        benchmark = self.benchmark
        templates = benchmark.templates
        try:
            template_to_conf_dict = PickleHandler.load(
                cache_header, f"{len(templates)}x{n_data_per_template}.pkl"
            )
            assert isinstance(template_to_conf_dict, Dict)
            logger.debug("template_to_conf_dict loaded")
        except BaseException:
            logger.debug("template_to_conf_dict not found, generating...")
            template_to_conf_dict = {
                template: spark_conf.get_lhs_configurations(
                    n_data_per_template, benchmark.template2id[template]
                )
                for template in templates
            }
            PickleHandler.save(
                template_to_conf_dict,
                cache_header,
                f"{len(templates)}x{n_data_per_template}.pkl",
            )
            logger.debug("template_to_conf_dict saved")
        return template_to_conf_dict

    def start_default(self, n_processes: int = 6, cluster_cores: int = 120) -> None:
        knob_sign = ",".join([str(k.default) for k in self.spark_conf.knob_list])
        cores = int(
            self.spark_conf.knob_dict_by_name["spark.executor.cores"].default
        ) * (
            int(self.spark_conf.knob_dict_by_name["spark.executor.instances"].default)
            + 1
        )
        default_header = f"{self.header}/default_conf"
        self._start(
            total=len(self.benchmark.templates),
            header=default_header,
            get_next=lambda index: (
                self.benchmark.templates[index],
                1,
                knob_sign,
                cores,
            ),
            cluster_cores=cluster_cores,
            n_processes=n_processes,
        )

    def start_lhs(
        self,
        n_data_per_template: int,
        cluster_cores: int = 120,
        seed: int = 42,
        n_processes: int = 6,
    ) -> None:
        templates = self.benchmark.templates
        template_to_conf_dict = self._get_lhs_conf_dict(n_data_per_template)
        query_matrix = QueryMatrix(
            templates=templates, n_data_per_template=n_data_per_template, seed=seed
        )
        total = query_matrix.total

        def prepare_lhs_i(index: int) -> Tuple[str, int, str, int]:
            template, qid = query_matrix.get_query_as_template_variant(index)
            conf_df = template_to_conf_dict[template].iloc[qid - 1]
            knob_sign = str(conf_df.name)
            cores = int(conf_df["spark.executor.cores"]) * (
                int(conf_df["spark.executor.instances"]) + 1
            )
            return template, qid, knob_sign, cores

        lhs_header = f"{self.header}/lhs_{len(templates)}x{n_data_per_template}"
        self._start(
            total=total,
            header=lhs_header,
            get_next=prepare_lhs_i,
            cluster_cores=cluster_cores,
            n_processes=n_processes,
        )

    def start_eval(
        self,
        eval_header: str,
        configurations: List[Dict],
        cluster_cores: int = 120,
        n_processes: int = 6,
    ) -> None:
        def prepare_eval_i(index: int) -> Tuple[str, int, str, int]:
            conf = configurations[index]
            (template, qid), configuration = list(conf.items())[0]
            knob_sign = self.spark_conf.conf2sign(
                configuration[knob_name] for knob_name in self.spark_conf.knob_names
            )
            cores = int(configuration["spark.executor.cores"]) * (
                int(configuration["spark.executor.instances"]) + 1
            )
            return template, qid, knob_sign, cores

        self._start(
            total=len(configurations),
            header=eval_header,
            get_next=prepare_eval_i,
            cluster_cores=cluster_cores,
            n_processes=n_processes,
        )
