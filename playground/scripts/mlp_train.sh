# mlp_train.sh

# Assigning positional parameters to variables
bm=$1
q_type=$2
model_choice=$3
nlayers=$4
hdim=$5
eps=$6
nworkers=$7
loss_weights=${8:-None}
gtn_n_layers=${9:-2}
gtn_n_heads=${10:-2}

# Default values for other variables
osize=32
lsize=8
vsize=16
lr=1e-3
bs=1024

if [ "$model_choice" = "gtn" ]; then
  command="python run_graph_gtn.py \
  --benchmark $bm \
  --q_type $q_type \
  --init_lr $lr \
  --min_lr 1e-7 \
  --epochs $eps \
  --batch_size $bs \
  --num_workers $nworkers \
  --lpe_size $lsize \
  --output_size $osize \
  --vec_size $vsize \
  --n_layers $nlayers \
  --hidden_dim $hdim \
  --gtn_n_layers $gtn_n_layers \
  --gtn_n_heads $gtn_n_heads \
  --pos_encoding_dim $lsize"
elif [ "$model_choice" = "raal" ]; then
  command="python run_graph_raal.py \
  --benchmark $bm \
  --q_type $q_type \
  --init_lr $lr \
  --min_lr 1e-7 \
  --epochs $eps \
  --batch_size $bs \
  --num_workers $nworkers \
  --lpe_size $lsize \
  --output_size $osize \
  --vec_size $vsize \
  --n_layers $nlayers \
  --hidden_dim $hdim \
  --gtn_n_layers $gtn_n_layers \
  --gtn_n_heads $gtn_n_heads \
  --pos_encoding_dim $lsize"
elif [ "$model_choice" = "qf" ]; then
  command="python run_graph_qf.py \
  --benchmark $bm \
  --q_type $q_type \
  --init_lr $lr \
  --min_lr 1e-7 \
  --epochs $eps \
  --batch_size $bs \
  --num_workers $nworkers \
  --lpe_size $lsize \
  --output_size $osize \
  --vec_size $vsize \
  --n_layers $nlayers \
  --hidden_dim $hdim \
  --gtn_n_layers $gtn_n_layers \
  --gtn_n_heads $gtn_n_heads \
  --pos_encoding_dim $lsize"
elif [ "$model_choice" = "avg" ]; then
  command="python run_graph_avg.py \
  --benchmark $bm \
  --q_type $q_type \
  --init_lr $lr \
  --min_lr 1e-7 \
  --epochs $eps \
  --batch_size $bs \
  --num_workers $nworkers \
  --lpe_size $lsize \
  --output_size $osize \
  --vec_size $vsize \
  --n_layers $nlayers \
  --hidden_dim $hdim"
else
   echo "not found"
fi

if [[ "$loss_weights" = "None" ]]; then
    $command
else
    command="$command --loss_weights $loss_weights"
    $command
fi
