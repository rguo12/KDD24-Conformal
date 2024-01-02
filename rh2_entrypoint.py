import subprocess

import torch
from rh2.sdk.env import get_rh2_env
from rh2.sdk.sdk_rh2_client import create_or_get_rh2_client

# get rh2 enviroment
env = get_rh2_env()

# get `epoch` param
dataset = env.params.dataset

# get input datasets
# train_dataset_id = env.inpputs.train_dataset.id
# test_dataset_id = env.inputs.test_dataset.id

# get output model
# output_model_id = env.outputs.output_model.id
# output_model_path = env.outputs.output_model.meta.hdfs_dir
output_folder = env.params.output_folder

print(f"output folder is {output_folder}")

# train code
cmd = f'''python3 run_syn.py --dataset={dataset}'''
print(f'cmd: {cmd}')
exit_code = subprocess.call(cmd, shell=True)

# copy generated model back to rh2
if exit_code == 0:
    print("done")
    subprocess.call(
        f'hadoop fs -copyFromLocal ./results/{dataset}/* {output_folder}/{dataset}/', shell=True)

    # client = create_or_get_rh2_client()
    # client.write_output_custom_meta(output_model_id, 'PYTORCH', {
    #                                 'version': torch.__version__})