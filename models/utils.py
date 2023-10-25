from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd

def split_data(data, n_folds, frac):
    X_train_list, T_train_list, Y_train_list = [], [], []
    X_calib_list, T_calib_list, Y_calib_list = [], [], []

    X = data.filter(like = 'X').values
    T = data['T'].values
    Y = data['Y'].values

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


def save_results(args, res, n_intervention, debug):
    res['n_intervention'] = n_intervention
    df = pd.DataFrame.from_dict(res, orient="index").transpose()

    if not os.path.exists(f'{args.save_path}/{args.dataset}_counterfactuals.csv'):
        df.to_csv(f'{args.save_path}/{args.dataset}_counterfactuals.csv')
    else:
        df.to_csv(f'{args.save_path}/{args.dataset}_counterfactuals.csv', mode='a', header=False)
    
    if debug:
        print(f"Weighted conformal prediction ({res['method']})")
        print("Number of intervention data", n_intervention)
        print('Coverage of Y(0)', res['coverage_0'])
        print('Interval width of Y(0)', res['interval_width_0'])
        print('Coverage of Y(1)', res['coverage_1'])
        print('Interval width of Y(1)', res['interval_width_1'])
        print("\n\n" + "=" * 20 + '\n\n')
    return


def preprocess(args):
    if os.path.exists(f'{args.save_path}/{args.dataset}_counterfactuals.csv'):
        os.remove(f'{args.save_path}/{args.dataset}_counterfactuals.csv')

    return args