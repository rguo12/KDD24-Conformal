import argparse
import os

dir_prefix = os.getcwd()

coat_params = {
    "train_path": dir_prefix + "/iDCF/data_process/coat/train.csv",
    "random_path": dir_prefix + "/iDCF/data_process/coat/random.csv",
    "user_feature_path": dir_prefix + "/iDCF/data_process/coat/user_feat_onehot.csv",
    "user_feature_label": dir_prefix + "/iDCF/data_process/coat/user_feat_label.csv",
    "dcf_A_hat_path": dir_prefix + "/iDCF/data_process/coat_wg_Afit/",
    "vae_path": dir_prefix + "/iDCF/data_process/coat_vae/",
    "ivae_path": dir_prefix + "/iDCF/data_process/coat_ivae/",
    # "test_ratio": 0.7,
    # "train_ratio": 0.8,
    "test_ratio": 0.25,
    "train_ratio": 0.5,
    "user_feature_dim": [2, 6, 3, 3, 2, 16, 13, 2],
    "threshold": 4.0,
    "min_val": 1.0,
    "max_val": 5.0,
    "batch_size": 1024,
    "beta_max": 1.,
    "name": "coat",
}

yahoo_params = {
    "train_path": dir_prefix + "/data_process/Yahoo_R3/train.csv",
    "random_path": dir_prefix + "/data_process/Yahoo_R3/random.csv",
    "user_feature_path": dir_prefix + "/data_process/Yahoo_R3/user_feat_onehot.csv",
    "user_feature_label": dir_prefix + "/data_process/Yahoo_R3/user_feat_label.csv",
    "dcf_A_hat_path": dir_prefix + "/data_process/R3_wg_Afit/",
    "ivae_path": dir_prefix + "/data_process/yahoo_ivae/",
    "vae_path": dir_prefix + "/data_process/yahoo_vae/",
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "user_feature_dim": [5, 5, 5, 5, 5, 5, 5],
    "threshold": 4.0,
    "min_val": 1.0,
    "max_val": 5.0,
    "batch_size": 512,
    "beta_max": 1.,
    "name": "yahoo"
}

kuai_rand_params = {
    "train_path": dir_prefix + "/data_process/kuai_rand/train.csv",
    "random_path": dir_prefix + "/data_process/kuai_rand/random.csv",
    "user_feature_path": dir_prefix + "/data_process/kuai_rand/user_feat_onehot.csv",
    "user_feature_label": dir_prefix + "/data_process/kuai_rand/user_feat_label.csv",
    "dcf_A_hat_path": dir_prefix + "/data_process/kuai_rand_wg_Afit/",
    "ivae_path": dir_prefix + "/data_process/kuai_rand_ivae/",
    "vae_path": dir_prefix + "/data_process/kuai_rand_vae/",
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "user_feature_dim": [9, 2, 2, 8, 9, 7, 8, 2, 7, 50, 1471, 33, 3, 118, 454, 7, 5, 4],
    "threshold": 0.9,
    "min_val": 0.0,
    "max_val": 5.0,
    "batch_size": 2048,
    "beta_max": 1.,
    "name": "kuai_rand"
}

simulation_params = {
    "train_path": dir_prefix + "/data_process/simulation/train.csv",
    "random_path": dir_prefix + "/data_process/simulation/random.csv",
    "user_feature_path": dir_prefix + "/data_process/simulation/user_feat_onehot.csv",
    "user_feature_label": dir_prefix + "/data_process/simulation/user_feat_label.csv",
    "dcf_A_hat_path": dir_prefix + "/data_process/sim_wg_Afit/",
    "ivae_path": dir_prefix + "/data_process/sim_ivae/",
    "vae_path": dir_prefix + "/data_process/sim_vae/",
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "user_feature_dim": [5, ],
    "threshold": 4.0,
    "min_val": 0.0,
    "max_val": 5.0,
    "batch_size": 1024,
    "beta_max": 1.,
    "name": "sim"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_prefix", type=str, default=dir_prefix)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--metric", type=str, default="mpe", help="[mpe, mse, ndcg]")
    parser.add_argument("--dataset", type=str, default="coat")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test_seed", action="store_true")
    parser.add_argument("--sim_suffix", type=str, default="")
    parser.add_argument("--key_name", type=str)
    
    parser.add_argument("--n_folds", type=int, default=1)
    parser.add_argument("--exact", type=bool, default=False)
    parser.add_argument("--dr_model", type=str, default="MLP")
    parser.add_argument("--standardize", type=bool, default=True)
    parser.add_argument("--method", type=str, default="naive")
    
    args = parser.parse_args()
    
    if args.dataset == "yahoo":
        data_params = yahoo_params
    elif args.dataset == "coat":
        data_params = coat_params
    elif args.dataset == "kuai_rand":
        data_params = kuai_rand_params
    elif args.dataset == "sim":
        data_params = simulation_params
        data_params["train_path"] = dir_prefix + "/data_process/simulation/train{}.csv".format(args.sim_suffix)
        data_params["random_path"] = dir_prefix + "/data_process/simulation/random{}.csv".format(args.sim_suffix)
        sr = args.sim_suffix.split("_")[2]
        tr = args.sim_suffix.split("_")[-1]
        data_params["ivae_path"] = dir_prefix + "/data_process/sim_ivae/sr_{}_tr_{}/".format(sr, tr)
        data_params["vae_path"] = dir_prefix + "/data_process/sim_vae/sr_{}_tr_{}/".format(sr, tr)
        data_params["dcf_A_hat_path"] = dir_prefix + "/data_process/sim_wg_Afit/sr_{}_tr_{}/".format(sr, tr)
    else:
        raise Exception("invalid dataset")
    setattr(args, "data_params", data_params)
    return args
