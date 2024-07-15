import os.path
import time
from argparse import ArgumentParser
from pathlib import Path

from udao_spark.model.mulitlabel_predictor import MultilabelPredictor  # type: ignore
from udao_spark.model.utils import wmape
from udao_spark.optimizer.utils import get_ag_meta
from udao_spark.utils.evaluation import get_ag_data
from udao_spark.utils.params import get_ag_parameters
from udao_trace.utils import JsonHandler


def get_params() -> ArgumentParser:
    parser = get_ag_parameters()
    # fmt: off
    parser.add_argument("--bm_gtn_model", type=str, default=None,
                        help="gtn of the model pretrained.")
    # fmt: on
    return parser


if __name__ == "__main__":
    params = get_params().parse_args()
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
    if bm_target != bm:
        ag_path = ag_meta["ag_path"] + f"_{bm_target}"
    else:
        ag_path = ag_meta["ag_path"]

    ag_path = ag_path + "/"
    if os.path.exists(ag_path):
        predictor = MultilabelPredictor.load(f"{ag_path}")
        print("model found and loadable at", ag_path)
    else:
        print("not found, start data preparation...")
        ret = get_ag_data(
            base_dir,
            bm,
            q_type,
            debug,
            graph_choice,
            weights_path=ag_meta["graph_weights_path"]
            if graph_choice != "none"
            else None,
            fold=params.fold,
            bm_target=bm_target,
        )
        train_data, val_data, test_data = ret["data"]
        ta, pw, objectives = ret["ta"], ret["pw"], ret["objectives"]
        if q_type.startswith("qs_"):
            objectives = list(filter(lambda x: x != "latency_s", objectives))
            train_data.drop(columns=["latency_s"], inplace=True)
            val_data.drop(columns=["latency_s"], inplace=True)
            test_data.drop(columns=["latency_s"], inplace=True)
        print("selected features:", train_data.columns)

        print("step 1: .fit()")
        start_time_step1 = time.perf_counter_ns()
        predictor = MultilabelPredictor(
            path=ag_path,
            labels=objectives,
            problem_types=["regression"] * len(objectives),
            eval_metrics=[wmape] * len(objectives),
            consider_labels_correlation=False,
        )
        time_dict_step1 = predictor.fit(
            train_data=train_data,
            # num_stack_levels=1,
            # num_bag_folds=4,
            # hyperparameters={
            #     "NN_TORCH": {},
            #     "GBM": {},
            #     "CAT": {},
            #     "XGB": {},
            #     "FASTAI": {},
            #     "RF": [
            #         {
            #             "criterion": "gini",
            #             "ag_args": {
            #                 "name_suffix": "Gini",
            #                 "problem_types": ["binary", "multiclass"],
            #             },
            #         },
            #         {
            #             "criterion": "entropy",
            #             "ag_args": {
            #                 "name_suffix": "Entr",
            #                 "problem_types": ["binary", "multiclass"],
            #             },
            #         },
            #         {
            #             "criterion": "squared_error",
            #             "ag_args": {
            #                 "name_suffix": "MSE",
            #                 "problem_types": ["regression", "quantile"],
            #             },
            #         },
            #     ],
            # },
            excluded_model_types=["KNN"],
            tuning_data=val_data,
            presets_lat=ag_sign,
            presets_io=ag_sign,  # we used to force this to be medium quality because
            # Shuffle Size / IO tends to have a much smaller error and we can save
            # inference time by using a lower quality model for IO
            use_bag_holdout=True,
            infer_limit=infer_limit,
            infer_limit_batch_size=infer_limit_batch_size,
            num_gpus=num_gpus,
            time_limit=None if time_limit is None else time_limit // len(objectives),
            ds_args={"memory_safe_fits": False},
            # set False to avoid memory issues when training io with high quality
        )
        dt1 = (time.perf_counter_ns() - start_time_step1) / 1e9

        print("step 2: .fit_weighted_ensemble()")
        start_time_step2 = time.perf_counter_ns()
        time_dict_step2 = {}
        for obj in predictor.predictors.keys():
            add_start_time = time.perf_counter_ns()
            models = predictor.get_predictor(obj).model_names(stack_name="core")
            return_models_po = predictor.get_predictor(obj).fit_weighted_ensemble(
                expand_pareto_frontier=True,
                name_suffix="PO",
            )
            return_models_fast_po = predictor.get_predictor(obj).fit_weighted_ensemble(
                base_models=[
                    m
                    for m in models
                    if "Large" not in m and "XT" not in m and "ExtraTree" not in m
                ],
                expand_pareto_frontier=True,
                name_suffix="FastPO",
            )

            time_dict_step2[obj] = (time.perf_counter_ns() - add_start_time) / 1e9

            print(f"get po-models from {obj}: {return_models_po}")
            print(f"get fast-po-models from {obj}: {return_models_fast_po}")
            print(
                f"ensemble models for {obj} including "
                f"{predictor.get_predictor(obj).model_names()}"
            )
        dt2 = (time.perf_counter_ns() - start_time_step2) / 1e9

        print(f"dt1: {dt1:.0f} s, dt2: {dt2:.0f} s")
        print("saving runtime to", ag_path)

        time_dict = {
            "dt1_s": dt1,
            "dt1_s_per_obj": time_dict_step1,
            "dt2_s": dt2,
            "dt2_s_per_obj": time_dict_step2,
        }
        JsonHandler.dump_to_file(time_dict, f"{ag_path}/runtime.json")
