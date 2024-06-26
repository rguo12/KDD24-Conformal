from torch import nn
from models.mf import MF
from torch.utils.data import Dataset, DataLoader
from utils import *
from conformal import *
# from ray.air import session
from argparser import *
from tune_script import *
from evaluator import Evaluator, mf_evaluate
from seeds import test_seeds

from datetime import datetime

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


# Generate current date in MM-DD format
current_date = datetime.now().strftime("%m_%d")
current_time = datetime.now().strftime("%H:%M:%S")

def train_eval(config):

    # metric = config["metric"]
    method = config["method"]

    print(f"running experiments for {method}")

    val_obs_loaders = []
    val_int_loaders = []
    test_int_loaders = []

    model_list = []
    
    dr_model_list = [] # save density models

    random_number = random.randint(1000, 9999)

    for i in range(config["n_folds"]):
        
        train_obs_loader, val_obs_loader, train_int_loader, val_int_loader, test_int_loader, evaluation_params, n_users, n_items, y_unique, InvP = construct_wcp_mf_dataloader(
            config, DEVICE)
        if method in ["exact", "inexact", "wcp", "wcp_ips"]:
            train_loader = train_obs_loader
            val_loader = val_obs_loader

        elif method == "naive":
            # train_int_loader, val_int_loader, test_int_loader, evaluation_params, n_users, n_items = construct_naive_mf_dataloader(config, DEVICE)
            # val_obs_loader = None
            train_loader = train_int_loader
            val_loader = val_int_loader
            
        else:
            raise ValueError(f"Unknown method {method}")
        
        seed_everything(config["seed"]+i) # make sure each fold is different

        # if val_obs_loader is not None:
        val_obs_loaders.append(val_obs_loader)
        val_int_loaders.append(val_int_loader)
        test_int_loaders.append(test_int_loader)

        # two quantile MF regression models, for upper/lower bound
        model = MF(n_users, n_items, y_unique, InvP, config["embedding_dim"]).to(DEVICE)
        # model_l = MF(n_users, n_items, config["embedding_dim"]).to(DEVICE)

        model_list.append(model)
        # model_l_list.append(model_l)

        # optimizer_u = torch.optim.Adam(params=model_u.parameters(), lr=config["lr_rate"], weight_decay=config["weight_decay"])
        optimizer = torch.optim.Adam(params=model.parameters(), 
                                     lr=config["lr_rate"], 
                                     weight_decay=config["weight_decay"])

        # loss_func = nn.MSELoss()
        # quantile_u = 0.95  # For upper bound model
        # quantile_l = 0.05  # For lower bound model

        # loss_func_u = PinballLoss(quantile_u)
        # loss_func_l = PinballLoss(quantile_l)

        loss_func = nn.MSELoss()

        evaluator = Evaluator("mse", patience_max=config["patience"])
        # evaluator_l = Evaluator("mpe", patience_max=config["patience"])

        # evaluator_test_coverage = Evaluator("test_coverage", patience_max=config["patience"])
        # evaluator_test_interval_width = Evaluator("test_interval_width", patience_max=config["patience"])

        for epoch in tqdm(range(config["epochs"])):

            # total_loss_u = 0
            total_loss = 0
            total_len = 0

            model.train()

            for index, (uid, iid, rating) in enumerate(train_loader):
                                
                uid, iid, rating = uid.to(DEVICE), iid.to(DEVICE), rating.float().to(DEVICE)

                # predict_u = model_u(uid, iid).view(-1)
                predict = model(uid, iid).view(-1)

                # loss_u = loss_func_u(predict_u, rating)
                loss = loss_func(predict, rating)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(rating)
                total_len += len(rating)

            evaluator.record_training(total_loss / total_len)

            model.eval()

            # select model by mpe
            validation_performance = mf_evaluate("mse", val_loader, model, device=DEVICE,
                                                params=evaluation_params, standardize=config["standardize"])
            
            # early stopping using model_u's performance only (not used in current implementation)
            early_stop = evaluator.record_val(validation_performance, model.state_dict())
            
            # TODO: only do conformal after training

            if not config["tune"]:
                pass
                
            if config["show_log"]:
                # evaluator_test_coverage.epoch_log(epoch)
                # evaluator_test_interval_width.epoch_log(epoch)
                evaluator.epoch_log(epoch)

            if early_stop:
                if config["show_log"]:
                    print("reach max patience {}, current epoch {}".format(evaluator.patience_max, epoch))
                break
        
        # we are doing this as the best model for u and l can be in different epochs, need to modify this later
        print("best val performance = {}".format(evaluator.get_val_best_performance()))
        # model_u.load_state_dict(evaluator_u.get_best_model())

        if method in ["naive", "wcp_ips"]:
            density_ratio_model = None

        elif method in ["exact", "inexact", "wcp"]:
            # train density ratio model after training MF model
            print(f"training density ratio model for {method}...")
            density_ratio_model = train_density_ratio(train_obs_loader, 
                                                      train_int_loader, 
                                                    model, method,
                                                    device=DEVICE,
                                                    dr_model=config["dr_model"],
                                                    mix_method=config["mix_method"],
                                                    dr_use_Y=config["dr_use_Y"])
            
            print("finished training density ratio model")
        dr_model_list.append(density_ratio_model)

    # evaluate with final model
    if method in ["exact", "inexact"]:
        # use val_obs as calibration, test_int as test

        ts_coverages, ts_inter_widths = mf_conf_eval_splitcp_mse(config,val_obs_loaders,
                                                             val_int_loaders, 
                                                            test_int_loaders, 
                                                            model_list,
                                                            dr_model_list,
                                                            method,
                                                            device=DEVICE, 
                                                            params=evaluation_params,
                                                            alpha=config["alpha"], 
                                                            standardize=config["standardize"],
                                                                dr_model=config["dr_model"], 
                                                                dataset=config["data_params"]["name"],
                                                                mix_method=config["mix_method"],
                                                                dr_use_Y=config["dr_use_Y"])
        
    elif method == "naive":
        ts_coverages, ts_inter_widths = mf_conf_eval_naive_mse(
                config, val_int_loaders, test_int_loaders, model_list, method,
                 device=DEVICE, params=evaluation_params, 
                 alpha=config["alpha"], 
                 standardize=config["standardize"], 
                 dataset=config["data_params"]["name"])


    elif method in ["wcp", "wcp_ips"]:
        ts_coverages, ts_inter_widths = mf_conf_eval_wcp_mse(config,
                val_obs_loaders, val_int_loaders, test_int_loaders, 
                model_list, dr_model_list, method,
                 device=DEVICE, params=evaluation_params, 
                 alpha=config["alpha"], standardize=config["standardize"], dr_model=config["dr_model"],
                   dataset=config["data_params"]["name"], mix_method=config["mix_method"])
    
    n_folds = len(ts_coverages)
    for i in range(n_folds):
        results = {
            "method": config["method"],
            "test_coverage": ts_coverages[i],
            "test_interval_width": ts_inter_widths[i],
            "mse": evaluator.get_val_best_performance(),
            "n_train_obs": config['n_train_obs'],
            "n_val_obs": config['n_val_obs'],
            "n_train_int": config['n_train_int'],
            "n_val_int": config['n_val_int'],
            "n_test_int": config['n_test_int'],
            }

        run_name = f"{current_date}/{current_time}_{config['method']}_{config['dr_model']}_seed_{config['seed']}"

        save_rec_results(config, results, run_name)
    
    # print(results)

    # if config["tune"]:
    #     session.report({
    #             "mpe": evaluator_u.get_val_best_performance(),
    #             "test_coverage": ts_coverages,
    #             "test_interval_width": ts_inter_widths
    #         })
        
        # if config["metric"] == "mse":
        #     session.report({
        #         "mse": evaluator.get_val_best_performance(),
        #         "test_mse": test_performance
        #     })
        # else:
        #     session.report({
        #         "ndcg": evaluator.get_val_best_performance(),
        #         "test_ndcg": test_performance[0],
        #         "test_recall": test_performance[1]
        #     })

    print("test coverage: {}, interval width: {}, best val mse: {}".format(
        ts_coverages, ts_inter_widths, evaluator.get_val_best_performance()))

if __name__ == '__main__':
    args = parse_args()
    model_name = "mf"
    # if args.tune:
    #     config = {
    #         "tune": True,
    #         "show_log": False,
    #         "patience": args.patience,
    #         "data_params": args.data_params,
    #         "metric": args.metric,
    #         "batch_size": args.data_params["batch_size"],
    #         "lr_rate": tune.grid_search([5e-5, 1e-5, 1e-3, 5e-4, 1e-4]),
    #         # "lr_rate": 5e-4,
    #         "epochs": 100,
    #         "weight_decay": tune.grid_search([1e-5, 1e-6]),
    #         # "weight_decay": 1e-6,
    #         "embedding_dim": 64,
    #         "seed": args.seed,
    #         "topk": args.topk
    #     }
    #     name_suffix = ""
    #     if args.test_seed:
    #         name_suffix = "_seed"
    #         if args.data_params["name"] == "coat":
    #             lr = 5e-4
    #             wd = 1e-6
    #         elif args.data_params["name"] == "yahoo":
    #             lr = 5e-4
    #             wd = 1e-5
    #         elif args.data_params["name"] == "sim":
    #             r_list = args.sim_suffix.split("_")
    #             sr = eval(r_list[2])
    #             cr = eval(r_list[4])
    #             tr = eval(r_list[-1])
    #             param = read_best_params(model_name, args.key_name, sr, cr, tr)
    #             lr = param["lr"]
    #             wd = param["wd"]
    #         elif args.data_params["name"] == "kuai_rand":
    #             lr = 5e-5
    #             wd = 1e-6
    #         config["lr_rate"] = lr
    #         config["weight_decay"] = wd
    #         config["seed"] = tune.grid_search(test_seeds)

    #     res_name = model_name + name_suffix
    #     if args.data_params["name"] == "sim":
    #         res_name = res_name + args.sim_suffix
    #     tune_param_rating(train_eval, config, args, res_name)
        
    # else:

    sample_config = {
        "metric": args.metric,
        "data_params": args.data_params,
        "tune": False,
        "show_log": True,
        "patience": args.patience,
        "lr_rate": 5e-4,
        "weight_decay": 1e-6,
        "epochs": args.epochs,
        "batch_size": args.data_params["batch_size"],
        "embedding_dim": args.embedding_dim,
        "topk": args.topk,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "dr_model":args.dr_model,
        "standardize":args.standardize,
        "method": args.method,
        "save_path": args.save_path,
        "alpha": args.alpha,
        "mix_method": args.mix_method,
        "dr_use_Y": args.dr_use_Y,
    }

    sample_config["data_params"]["test_ratio"] = args.test_ratio
    sample_config["data_params"]["val_ratio"] = args.val_ratio
    sample_config["data_params"]["train_ratio"] = args.train_ratio
    
    sample_config["data_params"]["obs_train_ratio"] = args.obs_train_ratio
    sample_config["data_params"]["obs_val_ratio"] = args.obs_val_ratio
    sample_config["data_params"]["obs_test_ratio"] = 1 - args.obs_train_ratio - args.obs_val_ratio


    train_eval(sample_config)
