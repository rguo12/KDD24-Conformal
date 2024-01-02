import bytedrh2
from bytedrh2.job import RunConfig
from bytedrh2.proto.core_pb2 import ArnoldConfig, ResourceRequirement
MERLIN_JOB_URL = " https://ml.bytedance.net/development/instance/jobs"
import json
# Load the credentials JSON file
"""
The user must have a credentials JSON file, named 'my_credentials.json' in this directory of the following form:
{
    "user_name": "xxx", 
    "access_token": "yyy", 
    "job_name": "zzz",
    "hdfs_user_name": "qqq"
}
"""

############################ Parameters to set ####################################
# run_probing = False
DATASET = "ihdp"
N_EST = 10
MODEL_NAME = "GBM"
DR_MODEL_NAME = "MLP"
TAG = f"_{MODEL_NAME}_{N_EST}_{}" + N_EST
STEP_BY_STEP = "True"
CORRECT_ANSWER = "True" # ---> will be ignored if running probing
# if run_probing:
RESULTS_DIR = f"./probing_models_{DATASET}_SBS_{STEP_BY_STEP}_{TAG}"
# else:
    # RESULTS_DIR = f"./results_{DATASET}_SBS_{STEP_BY_STEP}_{TAG}_correct_{CORRECT_ANSWER}"

N_CPU = 50
N_GPU = 0
env_vars = {"CORRECT_ANSWER": CORRECT_ANSWER}
FP16 = "False"
###### Parameters for linear_probing scripts -> WILL BE IGNORED IF `run_probing = False`
HDFS_RESULTS_PATH = "/home/byte_ailab_litg/user/muhammadtaufiq/results_hallueval_SBS_False_v5"
seeds = 1
tokens = 200
###### Parameters for get_representation scripts -> WILL BE IGNORED IF `run_probing = True`
n_runs_total = 50
###################################################################################

with open("my_credentials.json", "r") as file:
    data = json.load(file)
bytedrh2.authenticate(
    host="rh2.bytedance.net",
    user_name=data["user_name"],
    access_token=data["access_token"],
)
def createJob(params, env_vars={}):
    entrypointFullScript = "python3 main.py"
    env_vars.update({
        "ARNOLD_HACK_REGION": "NO_AWS",
        "HDFS_USERNAME": data["hdfs_user_name"],
        "RESULTS_DIR": RESULTS_DIR,
        "DATASET": DATASET,
        "STEP_BY_STEP": STEP_BY_STEP,
        "RUN_NO": str(params["run_no"]),
        "RUN_PROBING": str(run_probing),
        "HDFS_RESULTS_PATH": HDFS_RESULTS_PATH,
        "HDFS_RESULTS_DIR": HDFS_RESULTS_PATH.split("/")[-1],
        "RUN_PROBING": str(run_probing),
        "FP16": FP16,
        "MODEL_SIZE": MODEL_SIZE
    })  # Hack given by oncall to fix occasional machine failing to start
    qwe = {
        "cpu": N_CPU,
        "gpu": N_GPU,
        "gpuv": "NONE",
        "memory": 50120,
    }
    arnold_conf_role = ArnoldConfig.Role(name="worker", num=1, **qwe)
    resource = ResourceRequirement(
        arnold_config=ArnoldConfig(
            group_ids=[402],
            cluster_id=20,
            roles=[arnold_conf_role],
        ),
        backend="ARNOLD",
        # spark_app_conf=spark_app_conf
    )
    job = RunConfig(
        job_def_name=data["job_name"],
        entrypoint_args=entrypointFullScript,
        resource=resource,
        params=params,  # params here
        envs_list=env_vars,
    ).launch()
    return job
jobs = []
if run_probing:
    params = {"n_runs_total": -1, "run_no": -1}
    for i in range(seeds):
        for j in range(0, tokens, 2):
            env_vars.update({"SEED": str(i), "TOKEN": str(j)})
            createJob(params, env_vars)
else:
    for i in range(n_runs_total):
        params = {"n_runs_total": n_runs_total, "run_no": i}
        job = createJob(params, env_vars)