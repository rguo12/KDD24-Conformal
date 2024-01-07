import subprocess
import os

import torch
from rh2.sdk.env import get_rh2_env
from rh2.sdk.sdk_rh2_client import create_or_get_rh2_client

# get rh2 enviroment
env = get_rh2_env()


# local_data_path = f"/opt/tiger/causal_TCP/data/{dataset}"
# if not os.path.isdir(local_data_path):
#     os.mkdir(local_data_path)

# get input datasets
# train_dataset_id = env.inpputs.train_dataset.id
# test_dataset_id = env.inputs.test_dataset.id

# get output model
# output_model_id = env.outputs.output_model.id
# output_model_path = env.outputs.output_model.meta.hdfs_dir


dataset = env.params.dataset
output_folder = env.params.output_folder

n_inter_max = env.params.n_inter_max
n_obs = env.params.n_obs
# beta_u = env.params.beta_u
n_estimators= env.params.n_estimators
base_learner = env.params.base_learner
density_ratio_model = env.params.density_ratio_model
quantile_regression = env.params.quantile_regression
# n_Y_bins = env.params.n_Y_bins

# local_save_path = "/mnt/bn/confrank2/causal_TCP/results/"
subprocess.call('''cd causal_TCP''', shell=True)

local_save_path = "./results/"
if not os.path.exists(local_save_path):
    os.mkdir(local_save_path)
local_save_path = os.path.join(local_save_path,dataset)
if not os.path.exists(local_save_path):
    os.mkdir(local_save_path)

print(f"output folder is {output_folder}")

# train code

# cmd = f'''python3 run_syn.py --dataset={dataset} --save_path={local_save_path}'''
cmd = f'''python3 run_syn.py --dataset={dataset} \
        --save_path={local_save_path} \
        --n_inter_max={n_inter_max} \
        --n_obs={n_obs} \
        --n_estimators={n_estimators} \
        --base_learner={base_learner} \
        --density_ratio_model={density_ratio_model} \
        --quantile_regression={quantile_regression} \
        '''

print(f'cmd: {cmd}')
exit_code = subprocess.call(cmd, shell=True)

# copy generated model back to rh2
if exit_code == 0:
    print("done")
    # subprocess.call(f'hadoop fs -copyFromLocal /opt/tiger/causal_TCP/results/{dataset}/* {output_folder}/{dataset}/', shell=True)
    subprocess.call(f'hdfs dfs -put -f /opt/tiger/causal_TCP/results/{dataset}/* {output_folder}/{dataset}/', shell=True)
    
    # client = create_or_get_rh2_client()
    # client.write_output_custom_meta(dataset, local_save_path, output_folder, 'PYTORCH', {
    #                                 'version': torch.__version__})

