dataset="coat"
epochs=100
dr_model="DR"


methods=("inexact" "exact" "wcp_ips" "naive")


seeds=($(seq 1234 1238))
# seeds=(1234)
train_ratio=$1
val_ratio=$2
dr_use_Y=$3 # true or false
mix_method="product"

# not tuning
test_ratio=0.6
obs_train_ratio=0.8
obs_val_ratio=0.199
alpha=0.1
embedding_dim=64


for seed in "${seeds[@]}"
do
    for method in "${methods[@]}"
    do
        if [ "$condition" = true ]; then
            launch --gpu=1 --cpu=20 -- python3 iDCF/conf_mse_MF.py \
            --dr_use_Y \
            --mix_method $mix_method \
            --embedding_dim $embedding_dim \
            --alpha $alpha \
            --epochs $epochs \
            --method $method \
            --dr_model $dr_model \
            --dataset $dataset \
            --test_ratio $test_ratio \
            --train_ratio $train_ratio \
            --val_ratio $val_ratio \
            --obs_train_ratio $obs_train_ratio \
            --obs_val_ratio $obs_val_ratio \
            --seed $seed &>> "./ad_hoc_logs/$(date +"%Y_%m_%d_%I_%M_%p")_${dataset}_${method}.log"
        else
            launch --gpu=1 --cpu=20 -- python3 iDCF/conf_mse_MF.py \
            --mix_method $mix_method \
            --embedding_dim $embedding_dim \
            --alpha $alpha \
            --epochs $epochs \
            --method $method \
            --dr_model $dr_model \
            --dataset $dataset \
            --test_ratio $test_ratio \
            --train_ratio $train_ratio \
            --val_ratio $val_ratio \
            --obs_train_ratio $obs_train_ratio \
            --obs_val_ratio $obs_val_ratio \
            --seed $seed &>> "./ad_hoc_logs/$(date +"%Y_%m_%d_%I_%M_%p")_${dataset}_${method}.log"
        fi
        
    done
done