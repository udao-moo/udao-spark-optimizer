bm=$1
sample_mode=$2
nc=$3
np=$4
selected=${5:-0}
subdir=${6:-"revision"} # used to be "divB_new_grids"

# Constructing the base command based on the value of bm
if [ "$bm" = "tpch" ]; then
    cmd="python compile_time_hierarchical_optimizer.py \
        --q_type qs_lqp_compile \
        --ag_model_qs_ana_latency \"WeightedEnsemble_L2_FULL\" \
        --ag_model_qs_io \"CatBoost\" \
        --graph_choice gtn \
        --infer_limit 1e-5 \
        --infer_limit_batch_size 10000 \
        --ag_time_limit 21600 \
        --ag_sign \"high_quality\" \
        --moo_algo \"hmooc%B\" \
        --n_c_samples $nc \
        --n_p_samples $np \
        --sample_mode $sample_mode \
        --save_data_header \"./output/\" \
        --clf_recall_xhold -1"
elif [ "$bm" = "tpcds" ]; then
    cmd="python compile_time_hierarchical_optimizer.py \
        --benchmark tpcds \
        --q_type qs_lqp_compile \
        --ag_model_qs_ana_latency \"WeightedEnsemble_L3_FULL\" \
        --ag_model_qs_io \"CatBoost\" \
        --graph_choice gtn \
        --infer_limit 1e-5 \
        --infer_limit_batch_size 10000 \
        --ag_time_limit 10800 \
        --ag_sign \"high_quality\" \
        --moo_algo \"hmooc%B\" \
        --n_c_samples $nc \
        --n_p_samples $np \
        --sample_mode $sample_mode \
        --save_data_header \"./output/\" \
        --clf_recall_xhold -1"
else
    echo "Invalid benchmark specified"
    exit 1
fi

# Adding --selected_features if selected is true (1)
if [ "$selected" -eq 1 ]; then
    cmd="$cmd --selected_features"
fi

# Running the constructed command
eval $cmd

# Constructing the base command based on the value of bm
fname=nc${nc}_np${np}
if [ "$selected" -eq 1 ]; then
    fname="selected_params_${fname}"
fi

if [ "$bm" = "tpch" ]; then
  for weights in "0.0_1.0" "0.9_0.1" "0.5_0.5" "0.1_0.9" "1.0_0.0"; do
    scp chenghao_conf_save/tpch100/hmooc%B_${sample_mode}/${fname}_${weights}.json \
    hex1@node1:~/chenghao/udao-spark-optimizer/playground/evaluations/tpch100/${subdir}/on_demand/${sample_mode}_${fname}_${weights}.json
    scp chenghao_conf_save/tpch100/pref_to_df_hmooc%B_${sample_mode}_${fname}_${weights}.pkl \
    hex1@node1:~/chenghao/udao-spark-optimizer/playground/evaluations/tpch100/${subdir}
  done
elif [ "$bm" = "tpcds" ]; then
  for weights in "0.0_1.0" "0.9_0.1" "0.5_0.5" "0.1_0.9" "1.0_0.0"; do
    scp chenghao_conf_save/tpcds100/hmooc%B_${sample_mode}/${fname}_${weights}.json \
    hex2@node7:~/chenghao/udao-spark-optimizer/playground/evaluations/tpcds100/${subdir}/on_demand/${sample_mode}_${fname}_${weights}.json
    scp chenghao_conf_save/tpcds100/pref_to_df_hmooc%B_${sample_mode}_${fname}_${weights}.pkl \
    hex2@node7:~/chenghao/udao-spark-optimizer/playground/evaluations/tpcds100/${subdir}
  done
else
    echo "Invalid benchmark specified"
    exit 1
fi
