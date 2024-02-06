
import argparse 

from data.datasets import *
from models.methods import run_conformal, weighted_conformal_prediction
from models import utils
from datetime import datetime
import random

def get_config():
    parser = argparse.ArgumentParser(
        description='Transductive Conformal Prediction')

    # Data settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='./debug_results')
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='cevae')

    parser.add_argument('--n_folds', type=int, default=2)
    parser.add_argument('--test_frac', type=float, default=0.02)

    parser.add_argument('--n_obs', type=int, default=10000)

    # ihdp hidden conf strength
    parser.add_argument('--HC', type=bool, default=True, help="if False, X=U, for debug")
    parser.add_argument('--conf_strength', type=float, default=0.8, help="randomzied treatment when = 0.5")
    parser.add_argument('--x_dim', type=int, default=5)

    # Model settings
    parser.add_argument('--cf_method', type=str, default='tcp')
    # parser.add_argument('--ite_method', type=str, default='naive')

    parser.add_argument('--dr_use_Y', type=int, default=0, help="0: not use Y, 1: use Y, 2: use pseudo label")

    parser.add_argument('--base_learner', type=str, default="GBM")
    parser.add_argument('--density_ratio_model', type=str, default="MLP")
    parser.add_argument('--n_estimators', type=int, default=50)
    parser.add_argument('--quantile_regression', type=bool, default=True, 
                        help="True for quantile regression, now only supports quantile regression")
    
    parser.add_argument('--plot_dist', type=bool, default=False, 
                        help="True for plotting P(X,Y) for calib and test")

    args = parser.parse_args()

    return args

def main(args):
    # Get the current time
    current_time = datetime.now()
    cur_date = current_time.strftime("%m-%d")
    cur_time = current_time.strftime("%H:%M:%S")
    
    # Generating a 4 digit random integer to avoid fn collision
    random_number = random.randint(1000, 9999)

    args = utils.preprocess(args)
    np.random.seed(args.seed)

    n_observation = args.n_obs
    n_intervention_list = [100, 250, 500, 750, 1000]

    print(n_intervention_list)

    alpha = 0.1
    test_frac = args.test_frac # n_observation * (1. - test_frac) is the real n_observation
    n_folds = args.n_folds
    err_scale = 0.1

    dr_use_Y = args.dr_use_Y

    for n_intervention in n_intervention_list:
            
        if args.dataset == 'cevae':
            df_o, df_i = generate_cevae_data(n_observation, n_intervention, 
                                            conf_strength=args.conf_strength, 
                                            d=args.x_dim, 
                                            err_scale=err_scale,
                                            hidden_conf=args.HC)
        elif args.dataset == 'ihdp':
            # as ihdp is a small dataset w. 740+ samples
            # we only allow the n_intervention to be no larger than 500
            if n_intervention > 500:
                print("n_intervention must be no larger than 500 for ihdp dataset")
                return
            
            df_o, df_i = IHDP_w_HC(n_intervention, args.seed, d=24,
                hidden_confounding=True, beta_u=args.conf_strength, 
                root="/mnt/bn/confrank2/causal_TCP/data/IHDP")

            n_observation = df_o.shape[0]

        else:
            raise ValueError('select a dataset from [synthetic]')

        n_obs_treated = df_o[df_o['T']==1].shape[0]
        n_obs_controlled = df_o[df_o['T']==0].shape[0]

        n_inter_treated = df_i[df_i['T']==1].shape[0]
        n_inter_controlled = df_i[df_i['T']==0].shape[0]

        utils.save_dataset_stats(args, cur_date, 
                       n_obs_treated, n_obs_controlled, n_inter_treated, n_inter_controlled)
        
        # naive baseline
        if 'naive' == args.cf_method:
            res_list = run_conformal(df_o,
                                df_i,
                                quantile_regression=True,
                                n_folds=n_folds, #controls calib
                                alpha=alpha,
                                test_frac=test_frac, #controls test
                                target="counterfactual",
                                cf_method = 'naive',
                                # ite_method=args.ite_method,
                                plot=args.plot_dist,
                                dataset=args.dataset)
            for res in res_list:
                utils.save_results(args, res, n_intervention, n_observation, cur_date, cur_time, random_number)

        if 'inexact' == args.cf_method:
            res_list = run_conformal(
                                df_o,
                                df_i,
                                quantile_regression=True,
                                n_folds=n_folds,
                                alpha=alpha,
                                test_frac=test_frac,
                                target="counterfactual",
                                cf_method = 'inexact',
                                # ite_method=args.ite_method,
                                dr_use_Y=dr_use_Y)
            
            for res in res_list:
                utils.save_results(args, res, n_intervention, n_observation, cur_date, cur_time, random_number)

        if 'exact' == args.cf_method:

            res_list = run_conformal(
                                df_o,
                                df_i,
                                quantile_regression=True,
                                n_folds=n_folds,
                                alpha=alpha,
                                test_frac=test_frac,
                                target="counterfactual",
                                cf_method = 'exact',
                                # ite_method=args.ite_method,
                                dr_use_Y=dr_use_Y)
            
            for res in res_list:
                utils.save_results(args, res, n_intervention, n_observation, cur_date, cur_time, random_number)

        if 'wcp' == args.cf_method:
            res_list = weighted_conformal_prediction(df_o, 
                                            quantile_regression=True, 
                                            alpha=alpha, 
                                            test_frac=test_frac,
                                            target="counterfactual",
                                            # ite_method=args.ite_method,
                                            )
            
            for res in res_list:
                utils.save_results(args, res, n_intervention, n_observation, cur_date, cur_time, random_number)

        if 'tcp' == args.cf_method:
            res_list = run_conformal(df_o,
                                df_i,
                                quantile_regression=args.quantile_regression,
                                n_folds=n_folds,
                                alpha=alpha,
                                test_frac=test_frac,
                                target="counterfactual",
                                cf_method = 'tcp',
                                # ite_method=args.ite_method,
                                density_ratio_model=args.density_ratio_model,
                                base_learner=args.base_learner,
                                n_estimators=args.n_estimators,
                                dr_use_Y=dr_use_Y)
            
            for res in res_list:
                utils.save_results(args, res, n_intervention, n_observation, cur_date, cur_time, random_number)

            
        print(f"n_obs_treated: {n_obs_treated}, n_obs_controlled: {n_obs_controlled}, n_inter_treated: {n_inter_treated}, n_inter_controlled: {n_inter_controlled}")
    
    pause = True


if __name__ == '__main__':
    args = get_config()
    main(args)