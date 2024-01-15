from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
import time
import random

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def split_data(data, n_folds:int):
    """_summary_

    Args:
        data (_type_): _description_
        n_folds (int): number of folds for train/calib split
        frac (_type_): not used

    Returns:
        _type_: _description_
    """
    X_train_list, T_train_list, Y_train_list = [], [], []
    X_calib_list, T_calib_list, Y_calib_list = [], [], []

    X = data.filter(like = 'X').values
    T = data['T'].values
    Y = data['Y'].values

    # use this to maintain the same P(T) for tr and calib
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_index_list, calib_index_list = [], []
    for train_index, calib_index in skf.split(X, T):
        train_index_list.append(train_index)
        calib_index_list.append(calib_index)

    for i in range(n_folds):
        train_index = train_index_list[i]
        calib_index = calib_index_list[i]
        X_train, X_calib = X[train_index, :], X[calib_index, :]
        T_train, T_calib = T[train_index], T[calib_index]
        Y_train, Y_calib = Y[train_index], Y[calib_index]
        
        X_train_list.append(X_train)
        T_train_list.append(T_train)
        Y_train_list.append(Y_train)
        
        X_calib_list.append(X_calib)
        T_calib_list.append(T_calib)
        Y_calib_list.append(Y_calib)
        
    return train_index_list, X_train_list, T_train_list, Y_train_list, calib_index_list, X_calib_list, T_calib_list, Y_calib_list

def weighted_transductive_conformal(alpha, weights_train, weights_test, scores):
    """Weighted transductive conformal prediction

    Args:
        alpha (float): 1-alpha is the desired coverage
        weights_train (np.array (N_train,) ): weights for the training set
        weights_test (np.array (1, ) ): weights for the test set
        scores (np.array (N_train + 1, ) ): nonconformity scores for the training set

    Returns:
        offset (np.array (1, ) ): offset values for the test set
    """
    weights_train_sum = np.sum(weights_train)
    weights_train = weights_train / weights_train_sum
    q = (1 + weights_test / weights_train_sum) * (1 - alpha)
    q = np.minimum(q, 0.999)
    order = np.argsort(scores)
    scores = scores[order]
    weights = np.concatenate((weights_train, weights_test))
    weights = weights[order]
    cw = np.cumsum(weights)
    quantile_value = np.quantile(cw, q)
    index_quantile = np.argmax(cw >= quantile_value, axis=0)
    offset = scores[index_quantile]
    return offset


def weighted_conformal(alpha, weights_calib, weights_test, scores):
    """Weighted conformal prediction

    Args:
        alpha (float): 1-alpha is the desired coverage
        weights_calib (np.array (N_calib,) ): weights for the calibration set
        weights_test (np.array (N_test,) ): weights for the test set
        scores (np.array (N_calib, ) ): nonconformity scores for the calibration set

    Returns:
        offset (np.array (N_test, ) ): offset values for the test set
    """
    weights_calib_sum = np.sum(weights_calib)
    weights_calib = weights_calib / weights_calib_sum
    q = (1 + weights_test / weights_calib_sum) * (1 - alpha)
    q = np.minimum(q, 0.999)
    order = np.argsort(scores)
    scores = scores[order]
    weights_calib = weights_calib[order]
    cw = np.cumsum(weights_calib)
    cw_all = np.repeat(cw[:, None], len(weights_test), axis=1)
    quantile_value = np.quantile(cw_all, q)
    index_quantile = np.argmax(cw_all >= quantile_value[None,:], axis=0)
    offset = scores[index_quantile]
    return offset


def weights_and_scores(weight_fn, X_test, X_calib, Y_calib, Y_calib_hat_l, Y_calib_hat_u, model):
    weights_test = weight_fn(model, X_test)
    weights_calib = weight_fn(model, X_calib)
    scores  = np.maximum(Y_calib_hat_l - Y_calib, Y_calib - Y_calib_hat_u)
    return weights_calib, weights_test, scores


def standard_conformal(alpha, scores):
    q = (1 + 1. / len(scores)) * (1 - alpha)
    q = np.minimum(q, 0.999)
    order = np.argsort(scores)
    scores = scores[order]
    offset = np.quantile(scores, q)
    return offset


def save_dataset_stats(args, cur_time, 
                       n_obs_treated, n_obs_controlled, n_inter_treated, n_inter_controlled):

    res = {}
    res['n_obs_treated'] = n_obs_treated
    res['n_obs_controlled'] = n_obs_controlled
    res['n_inter_treated'] = n_inter_treated
    res['n_inter_controlled'] = n_inter_controlled

    df = pd.DataFrame.from_dict(res, orient="index").transpose()
    
    run_name = f"dataset_stats.csv"

    folder_name = os.path.join(args.save_path,args.dataset,cur_time) #local path
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    fn = os.path.join(folder_name,f'{run_name}.csv')

    print(f"saving results to {fn}")
    
    if not os.path.exists(fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        df.to_csv(fn)
    else:
        df.to_csv(fn, mode='a', header=False)

def save_results(args, res, n_intervention, n_observation, cur_time, random_number):
    res['n_intervention'] = n_intervention
    res['n_observation'] = n_observation
    res['conf_strength'] = args.conf_strength

    df = pd.DataFrame.from_dict(res, orient="index").transpose()
    
    run_name = f"{random_number}_{args.base_learner}_n_est_{args.n_estimators}_{args.density_ratio_model}_seed_{args.seed}"

    folder_name = os.path.join(args.save_path,args.dataset,cur_time) #local path
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    fn = os.path.join(folder_name,f'{run_name}.csv')

    print(f"saving results to {fn}")
    
    if not os.path.exists(fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        df.to_csv(fn)
    else:
        df.to_csv(fn, mode='a', header=False)
    
    if args.debug:
        print(f"Conformal prediction ({res['method']})")
        print("Number of intervention data", n_intervention)
        print("Number of observation data", n_observation)
        print('Coverage of Y(0)', res['coverage_0'])
        print('Interval width of Y(0)', res['interval_width_0'])
        print('Coverage of Y(1)', res['coverage_1'])
        print('Interval width of Y(1)', res['interval_width_1'])
        print("\n\n" + "=" * 20 + '\n\n')
    return


def preprocess(args):
    if os.path.exists(f'{args.save_path}/{args.dataset}_counterfactuals_{args.seed}.csv'):
        os.remove(f'{args.save_path}/{args.dataset}_counterfactuals_{args.seed}.csv')
    if args.seed is None:
        args.seed = int(time.time())
    return args


def plot_tsne(X_calib, X_test, j, dataset='ihdp', T=0, fig_name:str="featdist"):

    # Combine the calibration and test data
    X_combined = np.vstack((X_calib, X_test))

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)

    # Split the data back into calibration and test sets
    X_calib_tsne = X_tsne[:len(X_calib)]
    X_test_tsne = X_tsne[len(X_calib):]

    n_test = len(X_test)
    n_calib = len(X_calib)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_calib_tsne[:, 0], X_calib_tsne[:, 1], label='Calibration Data', c='blue', alpha=0.7)
    plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], label='Test Data', c='red', alpha=0.7)
    plt.title(f't-SNE Plot n_calib:{n_calib}, n_test:{n_test}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    # Save the plot as feature_dist_j.png
    plt.savefig(f'figs/{dataset}/{fig_name}_T_{T}_split_{j}.png')
    # plt.show()

# Example usage:
# Replace X_calib_j, X_test, and feature_name with your actual data and feature name.
# plot_tsne(X_calib_j, X_test, 'Your_Feature_Name')
