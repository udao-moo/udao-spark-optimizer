import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

from udao_spark.utils.params import get_base_parser
from udao_trace.configuration import SparkConf
from udao_trace.utils import JsonHandler, ParquetHandler, PickleHandler


def get_parser() -> ArgumentParser:
    parser = get_base_parser()
    # fmt: off
    parser.add_argument("--lhs_conf_file", type=str,
                        default="../udao_trace_examples/spark_collector/tpch100"
                                "/cache/template_to_conf_dict/22x2273.pkl")
    parser.add_argument("--trace_df_header", type=str,
                        default="cache_and_ckp/tpch_22x2273",
                        help="Header for the trace")
    # fmt: on
    return parser


params = get_parser().parse_args()
bm = params.benchmark
base_dir = Path(__file__).parent
sc = SparkConf(str(base_dir / "assets/spark_configuration_aqe_on.json"))

all_confs = PickleHandler.load(
    os.path.dirname(params.lhs_conf_file), os.path.basename(params.lhs_conf_file)
)
if not isinstance(all_confs, dict):
    raise ValueError("lhs_conf_file should be a dictionary")
df_success = ParquetHandler.load(params.trace_df_header, "df_q_compile.parquet")
df_success.template = df_success.template.astype(str)
conf_columns = sc.knob_names
knob_columns = sc.knob_ids
valid_clf = {}
for i, df_q_all in all_confs.items():
    df_knob_succ = df_success[df_success.template == i][knob_columns]
    df_knob_all = pd.DataFrame(
        data=sc.deconstruct_configuration(df_q_all[conf_columns].values),
        columns=knob_columns,
        index=df_q_all.index,
    )
    n_succ = df_knob_succ.shape[0]
    n_all = df_knob_all.shape[0]

    if n_succ == n_all:
        print(f"template {i}: all_pass")
        continue
    if n_all - n_succ < 10:
        print(f"template {i}: less than 5 failed, not enough data to test a clf, skip")
        continue
    print(f"template {i} success: {n_succ} / {n_all}, start training a clf...")
    merged = pd.merge(
        df_knob_all,
        df_knob_succ,
        on=df_knob_succ.columns.tolist(),
        how="left",
        indicator=True,
    )
    merged["failure"] = merged["_merge"] != "both"
    merged.drop(columns=["_merge"], inplace=True)
    X = merged.drop(
        "failure", axis=1
    )  # This removes the target column, leaving only features
    y = merged["failure"]  # This is your target column
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    train_data = TabularDataset(train_set)
    test_data = TabularDataset(test_set)
    label = "failure"
    predictor = TabularPredictor(
        label=label,
        path=f"AutogluonModels/{bm}_fail_clf/template_{i}",
        problem_type="binary",
        learner_kwargs={"label_count_threshold": 10},
    ).fit(
        train_data,
        infer_limit=1e-5,
        infer_limit_batch_size=10000,
        included_model_types=["XGB", "CAT"],
        hyperparameters="toy",
        presets=["medium_quality", "optimize_for_deployment"],
        verbosity=1,
    )
    eval_stats = predictor.evaluate(test_data)
    valid_clf[i] = {
        "path": predictor.path,
        "eval_stats": eval_stats,
        "n_all": n_all,
        "n_succ": n_succ,
    }
    print(eval_stats)

JsonHandler.dump_to_file(valid_clf, f"assets/{bm}_valid_clf_meta.json", indent=2)
