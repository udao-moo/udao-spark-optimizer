Modeling and Optimization
---

## Prepare the model

Note our optimizer can support customized models structures. Here is the version used in the paper. We first jointly train a query embedder (GTN) and regressor (MLP), and retrain the regressor as a weighted ensemble model with the fix query embedder weights.

```bash
export UDAOPATH=path\to\udao-spark-optimizer
export PYTHONENV=$pwd

cd playground

# jointly train
# bash scripts/mlp_train.sh <bm> <q_type> <embedder> <mlp-layers> <mlp-nhidden> <epochs> <num-workers>
# examples for TPC-H (additional hyperparameter tuning needed for a new setup)
bash scripts/mlp_train.sh tpch q_all gtn 5 512 100 # for collapsed LQP
bash scripts/mlp_train.sh tpch qs_lqp_compile gtn 5 512 100 # for subQ
bash scripts/mlp_train.sh tpch qs_pqp_runtime gtn 5 512 100 # for QS
# examples for TPC-DS (additional hyperparameter tuning needed for a new setup)
bash scripts/mlp_train.sh tpcds q_all gtn 5 512 100 # for collapsed LQP
bash scripts/mlp_train.sh tpcds qs_lqp_compile gtn 5 512 100 # for subQ
bash scripts/mlp_train.sh tpcds qs_pqp_runtime gtn 5 512 100 # for QS

# train a weighted ensemble model (trained with different time limits: 10800 and 21600)
python train_ensemble_models.py --hp_choice tuned-0215 --q_type qs_pqp_runtime \
--ag_sign "high_quality" --benchmark tpch --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 10800
python train_ensemble_models.py --hp_choice tuned-0215 --q_type qs_pqp_runtime \
--ag_sign "high_quality" --benchmark tpcds --infer_limit 1e-5 --infer_limit_batch_size 10000 --ag_time_limit 10800
```

## Compile-time Optimization

Only a subQ model (`qs_lqp_compile`) will be used for the compile-time optimization.

```bash
# TPC-H
# we got lat-WMAPE: 0.134, io-WAMPE=0.027, ~100ms for 14K theta choices in one call (xput in the test set: 70K/s)
model1=WeightedEnsemble_L2_FULL
model2=CatBoost
python compile_time_hierarchical_optimizer.py \
--q_type qs_lqp_compile \
--ag_model_qs_ana_latency $model1 \
--ag_model_qs_io $model2 \
--graph_choice gtn \
--infer_limit 1e-5 \
--infer_limit_batch_size 10000 \
--ag_time_limit 21600 \
--ag_sign "high_quality" \
--moo_algo "div_and_conq_moo%B" \
--n_c_samples 128 \
--n_p_samples 128 \
--clf_recall_xhold -1 # set to -1 to enable all trained classifiers

# TPC-DS
model1=WeightedEnsemble_L3_FULL
model2=CatBoost
python compile_time_hierarchical_optimizer.py \
--benchmark tpcds \
--q_type qs_lqp_compile \
--ag_model_qs_ana_latency $model1 \
--ag_model_qs_io $model2 \
--graph_choice gtn \
--infer_limit 1e-5 \
--infer_limit_batch_size 10000 \
--ag_time_limit 10800 \
--ag_sign "high_quality" \
--moo_algo "div_and_conq_moo%B" \
--n_c_samples 128 \
--n_p_samples 128 \
--clf_recall_xhold -1 # set to -1 to enable all trained classifiers
```

# Runtime Optimization

Use the LQP collapsed model (`q_all`) and the QS model (`qs_pqp_runtime`) for runtime optimization.

```bash
# TPC=H
python runtime_optimizer.py \
--benchmark tpch \
--graph_choice gtn \
--n_samples 3000 \
--ag_sign_q high_quality \
--infer_limit_q 1e-5 \
--infer_limit_batch_size_q 10000 \
--ag_model_q_latency WeightedEnsemble_L3_FULL \
--ag_model_q_io RandomForestMSE \
--ag_time_limit_q 10800 \
--ag_sign_qs medium_quality \
--infer_limit_qs 1e-5 \
--infer_limit_batch_size_qs 10000 \
--ag_model_qs_ana_latency WeightedEnsemble_L2PO_Pareto1 \
--ag_model_qs_io ExtraTreesMSE \
--verbose

# TPC-DS
python runtime_optimizer.py \
--benchmark tpcds \
--graph_choice gtn \
--n_samples 3000 \
--ag_sign_q high_quality \
--infer_limit_q 1e-5 \
--infer_limit_batch_size_q 10000 \
--ag_model_q_latency WeightedEnsemble_L2_FULL \
--ag_model_q_io CatBoost \
--ag_time_limit_q 10800 \
--ag_sign_qs high_quality \
--infer_limit_qs 1e-5 \
--infer_limit_batch_size_qs 10000 \
--ag_time_limit_qs 10800 \
--ag_model_qs_ana_latency WeightedEnsemble_L2_FULL \
--ag_model_qs_io RandomForestMSE \
--verbose
```
