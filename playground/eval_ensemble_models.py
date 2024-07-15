import os.path
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.optimizer.utils import get_ag_meta, get_lb_dict
from udao_spark.utils.collaborators import get_data_sign
from udao_spark.utils.evaluation import get_ag_data, get_ag_pred_objs
from udao_spark.utils.params import get_ag_parameters


def get_parser() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--ag_model_q_latency", type=str, default=None,
                        help="specific model name for AG for Q_R latency")
    parser.add_argument("--ag_model_q_io", type=str, default=None,
                        help="specific model name for AG for Q_R IO")
    parser.add_argument("--force", action="store_true",
                        help="Enable forcing running results")
    parser.add_argument("--bm_gtn_model", type=str, default=None,
                        help="gtn of the model pretrained.")
    parser.add_argument("--optimal", action="store_true",)
    parser.add_argument("--plus-tpl", action="store_true",)
    # fmt: on
    return parser


def find_best_model(meta: Dict, obj: str) -> str:
    lb_dict = get_lb_dict(
        ag_meta=meta["ag_meta"],
        ag_data=meta["ag_data"],
        weights_head=meta["weights_head"],
        lb_cache_name=meta["lb_cache_name"],
        plus_tpl=meta["plus_tpl"],
        q_type=meta["q_type"],
        split=meta["split"],
    )

    return (
        lb_dict[obj]
        .sort_values(["score_val", "model"], ascending=[False, True])
        .head(1)
        .model.values[0]
    )


if __name__ == "__main__":
    params = get_parser().parse_args()

    bm, q_type, debug = params.benchmark, params.q_type, params.debug
    hp_choice, graph_choice = params.hp_choice, params.graph_choice
    num_gpus, ag_sign = params.num_gpus, params.ag_sign
    infer_limit = params.infer_limit
    infer_limit_batch_size = params.infer_limit_batch_size
    time_limit = params.ag_time_limit
    base_dir = Path(__file__).parent
    ag_meta = get_ag_meta(
        bm,
        hp_choice,
        graph_choice,
        q_type,
        ag_sign,
        infer_limit,
        infer_limit_batch_size,
        time_limit,
        fold=params.fold,
    )
    bm_target = params.bm_gtn_model or bm
    ag_path = ag_meta["ag_path"] + ("" if bm == bm_target else f"_{bm_target}") + "/"

    ag_data = get_ag_data(
        base_dir,
        bm,
        q_type,
        debug,
        graph_choice,
        weights_path=ag_meta["graph_weights_path"] if graph_choice != "none" else None,
        fold=params.fold,
        bm_target=bm_target,
    )
    train_data, val_data, test_data = ag_data["data"]
    ta, pw, objectives = ag_data["ta"], ag_data["pw"], ag_data["objectives"]
    if q_type.startswith("qs_"):
        objectives = list(filter(lambda x: x != "latency_s", objectives))
        train_data.drop(columns=["latency_s"], inplace=True)
        val_data.drop(columns=["latency_s"], inplace=True)
        test_data.drop(columns=["latency_s"], inplace=True)
    print("selected features:", train_data.columns)

    if os.path.exists(ag_path):
        predictor = MultilabelPredictor.load(f"{ag_path}")
        print("loaded predictor from", ag_path)
    else:
        raise Exception("run train_ensemble_models.py first")

    lat_obj_name = "ana_latency_s" if q_type.startswith("qs") else "latency_s"

    if params.optimal:
        weights_head = (
            os.path.dirname(ag_meta["graph_weights_path"])
            if graph_choice != "none"
            else f"cache_and_ckp/{bm}_{get_data_sign(bm, False)}/{q_type}/none"
        )
        mysign = ag_sign
        if infer_limit is not None:
            mysign += f"-{infer_limit}-{infer_limit_batch_size}"
        if time_limit is not None:
            mysign += f"-{time_limit}"
        if params.plus_tpl:
            mysign += "-plus_tpl"
        if params.fold is not None:
            mysign += f"-{params.fold}"

        ag_model_dict = {}
        for split in ["val", "test"]:
            lb_cache_name = f"lb_{bm}_{q_type}_{split}_{mysign}.pkl"
            meta = {
                "ag_meta": ag_meta,
                "ag_data": ag_data,
                "weights_head": weights_head,
                "lb_cache_name": lb_cache_name,
                "plus_tpl": params.plus_tpl,
                "q_type": q_type,
                "split": split,
            }
            ag_model_dict[split] = {
                lat_obj_name: find_best_model(meta, "lat"),
                "io_mb": find_best_model(meta, "io"),
            }
        ag_model = ag_model_dict["val"]
    else:
        ag_model = {
            lat_obj_name: params.ag_model_q_latency,
            "io_mb": params.ag_model_q_io,
        }

    objs_true, objs_pred, dt_s, throughput, metrics = get_ag_pred_objs(
        base_dir,
        bm,
        q_type,
        debug,
        graph_choice,
        split="test",
        ag_meta=ag_meta,
        fold=params.fold,
        force=params.force,
        ag_model=ag_model,
        bm_target=bm_target,
        xfer_gtn_only=True,
        plus_tpl=params.plus_tpl,
    )

    print(f"metrics: {metrics}, throughput (regr only): {throughput} K/s")

    if q_type.startswith("qs"):
        m1, m2 = metrics["ana_latency_s"], metrics["io_mb"]
    else:
        m1, m2 = metrics["latency_s"], metrics["io_mb"]
    print("-" * 20)
    print(
        "{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & "
        "{:.3f} & {:.3f} & {:.3f} & {:.0f} \\\\".format(
            m1["wmape"],
            m1["p50_wape"],
            m1["p90_wape"],
            m1["corr"],
            m2["wmape"],
            m2["p50_wape"],
            m2["p90_wape"],
            m2["corr"],
            throughput,
        )
    )
    print()
