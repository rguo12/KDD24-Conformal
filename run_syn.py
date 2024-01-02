
import argparse 

from data.datasets import *
from models.methods import run_conformal, weighted_conformal_prediction
from models import utils

def get_config():
    parser = argparse.ArgumentParser(description='Transductive Conformal Prediction')

    # Data settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='ihdp')

    # Model settings
    parser.add_argument('--base_learner', type=str, default="GBM")
    parser.add_argument('--density_ratio_model', type=str, default="MLP")
    parser.add_argument('--n_estimators', type=int, default=20)
    parser.add_argument('--quantile_regression', type=bool, default=True, 
                        help="True for quantile regression, False for normal regression")

    args = parser.parse_args()

    return args

def main(args):
    args = utils.preprocess(args)
    np.random.seed(args.seed)
    n_observation = 10000
    n_intervention_list = np.arange(100, 1000, 100)
    d = 10

    alpha = 0.1
    test_frac = 0.5 # n_observation * (1. - test_frac) is the real n_observation
    n_folds = 5
    err_scale = 0.1

    # df_train, df_test = generate_lilei_hua_data()
    # _ = weighted_conformal_prediction([df_train, df_test], 
    #                                   metalearner="DR", 
    #                                   quantile_regression=True, 
    #                                   alpha=0.1, 
    #                                   test_frac=0.1)
    # df_o = [df_train, df_test]


    for n_intervention in n_intervention_list:
        
        if args.dataset == 'synthetic':
            df_o, df_i = generate_data(n_observation=n_observation,    
                                n_intervention=n_intervention,
                                d=d, 
                                gamma=0.5, 
                                alpha=alpha,
                                confouding=True)
            
        elif args.dataset == 'cevae':
            df_o, df_i = generate_cevae_data(n_observation, n_intervention, err_scale = err_scale)


        elif args.dataset == 'ihdp':
            # as ihdp is a small dataset w. 740+ samples
            # we only allow the n_intervention to be no larger than 500

            if n_intervention > 500:
                break
            df_o, df_i = IHDP_w_HC(n_intervention, args.seed, d=24,
              hidden_confounding=True, beta_u=None, root="data/IHDP")
        
        # naive baseline
        res = run_conformal(
                            df_o,
                            df_i,
                            quantile_regression=True,
                            n_folds=n_folds,
                            alpha=alpha,
                            test_frac=test_frac,
                            target="counterfactual",
                            method = 'naive')
        
        utils.save_results(args, res, n_intervention)


        res = run_conformal(
                            df_o,
                            df_i,
                            quantile_regression=True,
                            n_folds=n_folds,
                            alpha=alpha,
                            test_frac=test_frac,
                            target="counterfactual",
                            method = 'inexact')
        
        utils.save_results(args, res, n_intervention)

        res = run_conformal(
                            df_o,
                            df_i,
                            quantile_regression=True,
                            n_folds=n_folds,
                            alpha=alpha,
                            test_frac=test_frac,
                            target="counterfactual",
                            method = 'exact')
        
        utils.save_results(args, res, n_intervention)


        res = weighted_conformal_prediction(df_o, 
                                        quantile_regression=True, 
                                        alpha=alpha, 
                                        test_frac=test_frac,
                                        target="counterfactual",
                                        method='weighted CP')
        
        utils.save_results(args, res, n_intervention)

        res = run_conformal(
                            df_o,
                            df_i,
                            quantile_regression=args.quantile_regression, # QR not implemented yet
                            n_folds=n_folds,
                            alpha=alpha,
                            test_frac=test_frac,
                            target="counterfactual",
                            method = 'TCP',
                            density_ratio_model=args.density_ratio_model,
                            base_learner=args.base_learner,
                            n_estimators=args.n_estimators)
        
        utils.save_results(args, res, n_intervention)

        # coverage, average_interval_width, PEHE, conformity_scores = conformal_metalearner(df_o, 
        #                                                                                 metalearner="DR", 
        #                                                                                 quantile_regression=True, 
        #                                                                                 alpha=0.1, 
        #                                                                                 test_frac=0.1)

    pause = True
    return


if __name__ == '__main__':
    args = get_config()
    main(args)