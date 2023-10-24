import jax
import jax.numpy as jnp
import argparse 

from data.datasets import *
from models.methods import *


def get_config():
    parser = argparse.ArgumentParser(description='Transductive Conformal Prediction')

    # Data settings
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()
    return args

def main(args):
    n_observation = 1000
    n_intervention = 100
    d = 10

    synthetic_setups = dict({"A": 1, "B": 0})
    setup = 'A'
    alpha = 0.1
    test_frac=0.1
    # df_train, df_test = generate_lilei_hua_data()
    # _ = weighted_conformal_prediction([df_train, df_test], 
    #                                   metalearner="DR", 
    #                                   quantile_regression=True, 
    #                                   alpha=0.1, 
    #                                   test_frac=0.1)
    # df_o = [df_train, df_test]
    rng_key = jax.random.PRNGKey(args.seed)
    df_o, df_i = generate_data(rng_key=rng_key,
                               n_observation=n_observation,    
                                n_intervention=n_intervention,
                                d=d, 
                                gamma=0.5, 
                                alpha=alpha,
                                confouding=True) 
    
    
    coverage_0, coverage_1, interval_width_0, interval_width_1 = transductive_weighted_conformal(df_o,
                                        df_i,
                                        quantile_regression=True,
                                        alpha=alpha,
                                        test_frac=test_frac,
                                        method="counterfactual")
    print("Transductive weighted conformal prediction (Ours)")
    print('Coverage of Y(0)', coverage_0)
    print('Interval width of Y(0)', interval_width_0)
    print('Coverage of Y(1)', coverage_1)
    print('Interval width of Y(1)', interval_width_1)

    coverage_0, coverage_1, interval_width_0, interval_width_1 = weighted_conformal_prediction(df_o, 
                                      quantile_regression=True, 
                                      alpha=alpha, 
                                      test_frac=test_frac,
                                      method="counterfactual")
    print("\n\n" + "=" * 20 + '\n\n')
    print("Split weighted conformal prediction (Li Leihua)")
    print('Coverage of Y(0)', coverage_0)
    print('Interval width of Y(0)', interval_width_0)
    print('Coverage of Y(1)', coverage_1)
    print('Interval width of Y(1)', interval_width_1)
    
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