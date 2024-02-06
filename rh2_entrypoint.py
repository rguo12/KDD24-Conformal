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

x_dim = env.params.x_dim
conf_strength = env.params.conf_strength
n_estimators= env.params.n_estimators
seed = env.params.seed # only run tcp on arnold

# n_inter_max = env.params.n_inter_max
# n_obs = env.params.n_obs
# # beta_u = env.params.beta_u
# base_learner = env.params.base_learner
# density_ratio_model = env.params.density_ratio_model
# quantile_regression = env.params.quantile_regression


# seed = env.params.seed
# n_Y_bins = env.params.n_Y_bins

# local_save_path = "/mnt/bn/confrank2/causal_TCP/results/"
subprocess.call('''cd causal_TCP''', shell=True)

local_save_path = "./results/"
if not os.path.exists(local_save_path):
    os.mkdir(local_save_path)
local_save_path_ = os.path.join(local_save_path,dataset)
if not os.path.exists(local_save_path_):
    os.mkdir(local_save_path_)

print(f"output folder is {output_folder}")

# train code

# cmd = f'''python3 run_syn.py --dataset={dataset} --save_path={local_save_path}'''
# for seed in range(1234,1239):
cmd = f'''bash run_tcp.sh {x_dim} {conf_strength} {n_estimators} {seed}'''
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

