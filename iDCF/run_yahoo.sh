dataset="yahoo"
epochs=100
dr_model="DR"


methods=("naive" "wcp_ips") # Updated to include both methods

# Uncomment and adjust the next line if you want to use a range of seeds
seeds=($(seq 1234 1238))

train_ratio=$1 # ratio of int data for tr
val_ratio=$2 # ratio of int data for cal
dr_use_Y=$3 # true or false
mix_method="product"

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
            launch --gpu=1 --cpu=30 -- python3 iDCF/conf_mse_MF.py \
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
            launch --gpu=1 --cpu=30 -- python3 iDCF/conf_mse_MF.py \
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