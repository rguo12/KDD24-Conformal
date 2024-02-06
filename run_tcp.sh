dataset="cevae"

n_folds=2
test_frac=0.02

# vars to modify
# methods=("wcp")
# methods=("exact" "inexact")
# methods=("inexact" "exact" "wcp" "naive")
# methods=("TCP")

dr_model="MLP"

n_obs=10000

dr_use_Y=1

x_dim=$1
conf_strength=$2
n_estimator=$3
seed=$4

methods=("tcp")


# each run considers n_int = (100 500 1000 5000)

for method in "${methods[@]}"
do
    python3  /opt/tiger/causal_TCP/run_syn.py \
    --cf_method $method \
    --density_ratio_model $dr_model \
    --n_obs $n_obs \
    --dr_use_Y $dr_use_Y \
    --conf_strength $conf_strength \
    --x_dim $x_dim \
    --dataset $dataset \
    --seed $seed \
    --n_estimator $n_estimator \
    --n_folds $n_folds \
    --save_path "./results" \
    --test_frac $test_frac
done