from itertools import chain
from typing import Any, Dict, Optional

import numpy as np

from udao_trace.configuration import SparkConf
from udao_trace.parser.spark_parser import drop_raw_plan, parse_conf
from udao_trace.utils import JsonHandler

from ...utils.constants import BETA, EPS, THETA_C, THETA_P
from ..utils import extract_compile_time_im


def extract_alpha(
    size_in_mb: float, rows_count: float, add_compile_tag: bool
) -> Dict[str, float]:
    suffix = "-compile" if add_compile_tag else ""
    return {
        f"IM-sizeInMB{suffix}": size_in_mb,
        f"IM-rowCount{suffix}": rows_count,
        f"IM-sizeInMB{suffix}-log": np.log(np.clip(size_in_mb, a_min=EPS, a_max=None)),
        f"IM-rowCount{suffix}-log": np.log(np.clip(rows_count, a_min=EPS, a_max=None)),
    }


def extract_alpha_plus(init_part_num: int) -> Dict[str, float]:
    return {
        "IM-init-part-num": float(init_part_num),
        "IM-init-part-num-log": np.log(np.clip(init_part_num, a_min=EPS, a_max=None)),
    }


def extract_beta(pd_dict: Dict) -> Dict[str, float]:
    pd = np.array(list(chain.from_iterable(pd_dict.values())))
    beta_ratios = [0.0, 0.0, 0.0]
    if pd.size > 0:
        mu, std, max_val, min_val = np.mean(pd), np.std(pd), np.max(pd), np.min(pd)
        beta_ratios = [std / mu, (max_val - mu) / mu, (max_val - min_val) / mu]
    beta_dict = {k: v for k, v in zip(BETA, beta_ratios)}
    return beta_dict


def extract_gamma(d: Dict) -> Dict[str, float]:
    ss_dict = {}
    if "RunningQueryStageSnapshot" in d:
        for k, v in d["RunningQueryStageSnapshot"].items():
            if isinstance(v, list):
                for tile, v_ in zip([0, 25, 50, 75, 100], v):
                    ss_dict[f"SS-{k}-{tile}tile"] = v_
            else:
                ss_dict[f"SS-{k}"] = v
    else:
        raise ValueError("No RunningQueryStageSnapshot in d")
    return ss_dict


def get_non_decision_inputs_for_q_compile(json_path: str) -> Dict[str, Any]:
    d = JsonHandler.load_json(json_path)["CompileTimeLQP"]
    alpha = extract_alpha(
        size_in_mb=d["IM"]["inputSizeInBytes"] / 1024 / 1024,
        rows_count=d["IM"]["inputRowCount"] * 1.0,
        add_compile_tag=False,
    )
    return {
        "lqp": JsonHandler.dump_to_string(drop_raw_plan(d["LQP"]), indent=None),
        **alpha,
    }


def get_non_decision_inputs_for_qs_compile(qs: Dict) -> Dict[str, Any]:
    qs_dict = qs["QSLogical"]
    qs_lqp = JsonHandler.dump_to_string(drop_raw_plan(qs_dict), indent=None)
    size_in_mb, rows_count = extract_compile_time_im(qs_lqp)
    alpha = extract_alpha(
        size_in_mb=size_in_mb, rows_count=rows_count, add_compile_tag=True
    )
    return {
        "qs_lqp": qs_lqp,
        **alpha,
    }


def get_non_decision_inputs_for_qs_compile_dict(
    json_path: str, is_oracle: bool = False
) -> Dict[str, Dict[str, Any]]:
    d = JsonHandler.load_json(json_path)["RuntimeQSs"]
    return {
        f"qs-{qs_id}": get_non_decision_inputs_for_qs_runtime(qs, is_lqp=True)
        if is_oracle
        else get_non_decision_inputs_for_qs_compile(qs)
        for qs_id, qs in d.items()
    }


def get_non_decision_inputs_for_q_runtime(d: Dict, sc: SparkConf) -> Dict[str, Any]:
    alpha = extract_alpha(
        size_in_mb=d["IM"]["inputSizeInBytes"] / 1024 / 1024,
        rows_count=d["IM"]["inputRowCount"] * 1.0,
        add_compile_tag=False,
    )
    beta = extract_beta(d["PD"])
    gamma = extract_gamma(d)
    conf = parse_conf(d["Configuration"])
    theta_np = sc.deconstruct_configuration(np.array([list(conf.values())]))
    theta = {k: v for k, v in zip(THETA_C, theta_np[0][: len(THETA_C)])}
    return {
        "lqp": JsonHandler.dump_to_string(drop_raw_plan(d["LQP"]), indent=None),
        **alpha,
        **beta,
        **gamma,
        **theta,
    }


def get_non_decision_inputs_for_qs_runtime(
    qs: Dict, is_lqp: bool, sc: Optional[SparkConf] = None
) -> Dict[str, Any]:
    if is_lqp:
        qs_plan = {
            "qs_lqp": JsonHandler.dump_to_string(
                drop_raw_plan(qs["QSLogical"]), indent=None
            )
        }
        theta = {}
    else:
        if sc is None:
            raise ValueError("sc cannot be None for runtime pqp")
        qs_plan = {
            "qs_pqp": JsonHandler.dump_to_string(
                drop_raw_plan(qs["QSPhysical"]), indent=None
            )
        }
        conf = parse_conf(qs["Configuration"])
        theta_np = sc.deconstruct_configuration(np.array([list(conf.values())]))
        theta = {
            k: v
            for k, v in zip(THETA_C + THETA_P, theta_np[0][: len(THETA_C + THETA_P)])
        }
        # include THETA_P to help detect failure.

    alpha = extract_alpha(
        size_in_mb=qs["IM"]["inputSizeInBytes"] / 1024 / 1024,
        rows_count=qs["IM"]["inputRowCount"] * 1.0,
        add_compile_tag=False,
    )
    alpha_plus = {} if is_lqp else extract_alpha_plus(qs["InitialPartitionNum"])
    beta = extract_beta(qs["PD"])
    gamma = extract_gamma(qs)
    return {
        **qs_plan,
        **alpha,
        **alpha_plus,
        **beta,
        **gamma,
        **theta,
    }
