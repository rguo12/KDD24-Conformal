from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def split_data(data, n_folds, frac):
    index_1_list, index_2_list = [], []
    X_1_list, T_1_list, Y_1_list = [], [], []
    X_2_list, T_2_list, Y_2_list = [], [], []
    for _ in range(n_folds):
        index_1 = np.random.permutation(data.index)[:int(frac * len(data.index))]
        index_2 = np.random.permutation(data.index)[int(frac * len(data.index)):]
        X_1, X_2 = data.loc[index_1].filter(like = 'X').values, data.loc[index_2].filter(like = 'X').values
        T_1, T_2 = data.loc[index_1]['T'].values, data.loc[index_2]['T'].values
        Y_1, Y_2 = data.loc[index_1]['Y'].values, data.loc[index_2]['Y'].values
        index_1_list.append(index_1)
        X_1_list.append(X_1)
        T_1_list.append(T_1)
        Y_1_list.append(Y_1)
        index_2_list.append(index_2)
        X_2_list.append(X_2)
        T_2_list.append(T_2)
        Y_2_list.append(Y_2)
    return [index_1_list, X_1_list, T_1_list, Y_1_list], [index_2_list, X_2_list, T_2_list, Y_2_list]

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
    q = np.minimum(q, 0.99)
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
    q = np.minimum(q, 0.99)
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
    q = (1 + len(scores)) * (1 - alpha)
    q = np.minimum(q, 0.99)
    order = np.argsort(scores)
    scores = scores[order]
    offset = np.quantile(scores, q)
    return offset
