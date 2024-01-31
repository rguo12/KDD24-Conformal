import bottleneck as bn
import numpy as np
import random
import os
# import ray
import json
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from sklearn.metrics import mean_squared_error, mean_pinball_loss
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import torch.nn as nn

from densratio import densratio
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor

from sklearn.metrics import accuracy_score

from utils import plot_vec_dist

base_learners_dict = dict({"GBM": GradientBoostingRegressor, 
                           "QRF": RandomForestQuantileRegressor})



def standard_conformal(alpha, scores):
    q = (1 + 1. / len(scores)) * (1 - alpha)
    q = np.minimum(q, 0.999)
    order = np.argsort(scores)
    scores = scores[order]
    offset = np.quantile(scores, q)
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

def get_density_ratio_data(data_loader, model, method, device="cpu", mix_method="concat"):
    # collect data
    labels = []
    embeddings = []
    for index, (uid, iid, rating) in enumerate(data_loader):
        uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
        U, I = model.get_embedding(uid,iid)
        if mix_method == "concat":
            ui_embedding = torch.cat([U,I],dim=1)
        elif mix_method == "sum":
            ui_embedding = U+I
        elif mix_method == "product":
            ui_embedding = U*I
        ui_embedding = ui_embedding.detach().cpu().numpy()
        embeddings.append(ui_embedding)
        labels.extend(rating.tolist())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels).reshape(-1,1)
    if method == "wcp":
        return embeddings
    elif method in ["exact", "inexact"]:
        D = np.concatenate([embeddings,labels],axis=1)
        return D
    else:
        raise ValueError("incorrect method")



def get_ips_weights(data_loader, model, method, device="cpu"):
    # collect ps for wcp_ips

    ips_weights = []

    for index, (uid, iid, rating) in enumerate(data_loader):
        uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
        batch_ips_weights = model.compute_ips_weights(uid,iid,rating)
        ips_weights.append(batch_ips_weights.detach().cpu().numpy())
    
    ips_weights = np.concatenate(ips_weights, axis=0)
    # labels = np.array(labels).reshape(-1,1)

    if method == "wcp_ips":
        return ips_weights
    
    else:
        raise ValueError("incorrect method")

def train_density_ratio(train_obs_loader, train_int_loader, model, method, 
                        device="cpu", dr_model="DR", mix_method="concat", 
                         dr_use_Y:bool = False):
    
    # dr_use_Y: whether use Y in density ratio estimation

    D_train_obs = get_density_ratio_data(train_obs_loader, model, method, device=device, mix_method=mix_method)
    D_train_int = get_density_ratio_data(train_int_loader, model, method, device=device, mix_method=mix_method)

    print(D_train_obs.shape)
    print(D_train_int.shape)

    if not dr_use_Y:
        D_train_obs = D_train_obs[:,:-1]
        D_train_int = D_train_int[:,:-1]

    print(D_train_obs.shape)
    print(D_train_int.shape)
    
    if dr_model=="DR":
        density_ratio_model = densratio(D_train_int, D_train_obs, verbose=False, alpha=0.01)

    elif dr_model == "MLP":
        density_ratio_model = MLPClassifier(random_state=42, max_iter=100)

        # Assigning labels
        Y_obs_mlp = np.ones(len(D_train_obs))  # Label 1 for observed class
        Y_inter_mlp = np.zeros(len(D_train_int))  # Label 0 for interventional class

        # Merging the datasets
        X_mlp = np.concatenate((D_train_obs, D_train_int))
        Y_mlp = np.concatenate((Y_obs_mlp, Y_inter_mlp))

        density_ratio_model.fit(X_mlp, Y_mlp)

        Y_mlp_pred = density_ratio_model.predict(X_mlp)
        acc = accuracy_score(Y_mlp, Y_mlp_pred)
        print(f"acc of the dr_model {acc}")

    return density_ratio_model


def mf_calib(data_loader, model_u, model_l, device="cpu", alpha=0.1, params=None, standardize=False):
    """
    compute quantile of nonconformity scores

    Args:
        data_loader (_type_): test data loader
        model_u (_type_): trained upper bound model
        model_l (_type_): trained lower bound model
        device (str, optional): _description_. Defaults to "cpu".
        alpha (float, optional): _description_. Defaults to 0.95.
    """

    # calibration

    scores_list = []
    for index, (uid, iid, rating) in enumerate(data_loader):
        uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
        predict_u = model_u.predict(uid, iid)
        predict_l = model_l.predict(uid, iid)
        if standardize:
            predict_u = params["min_val"] + predict_u * (params["max_val"] - params["min_val"])
            predict_l = params["min_val"] + predict_l * (params["max_val"] - params["min_val"])
                
        scores = torch.maximum(predict_l - rating,
                                rating - predict_u).detach().cpu().numpy()
        scores_list.append(scores)
    scores_list = np.concatenate(scores_list)
 
    return scores_list

def mf_calib_mse(data_loader, model, device="cpu", params=None, standardize=False):
    """
    compute quantile of nonconformity scores

    Args:
        data_loader (_type_): test data loader
        model_u (_type_): trained upper bound model
        model_l (_type_): trained lower bound model
        device (str, optional): _description_. Defaults to "cpu".
        alpha (float, optional): _description_. Defaults to 0.95.
    """

    # calibration

    scores_list = []
    for index, (uid, iid, rating) in enumerate(data_loader):
        uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
        predict = model.predict(uid, iid)
        if standardize:
            predict = params["min_val"] + predict * (params["max_val"] - params["min_val"])

        # TODO: check the scale of the rating and predict
        scores = torch.abs(predict - rating).detach().cpu().numpy()
        scores_list.append(scores)
    scores_list = np.concatenate(scores_list)
 
    return scores_list


# def mf_conf_eval_splitcp(cal_obs_loaders:list, cal_int_loaders:list, test_int_loaders:list, 
#                          model_u_list:list, model_l_list:list, dr_model_list:list,
#                         device:str="cpu", params=None, alpha:float=0.1, standardize:bool=True,
#                         base_learner:str="RF", n_estimators:int=10, exact:bool=False,
#                         dr_model:str="DR"):

#     """
#     do split CP (exact and inexact for quantile regression)
#     """
    
#     print("start conf eval for our split cp methods")

#     # first do weighted CP on calib

#     n_folds = len(cal_obs_loaders)

#     coverages = []
#     interval_widths = []
    
#     for i in range(n_folds):
#         model_u = model_u_list[i]
#         model_l = model_l_list[i]
#         density_model = dr_model_list[i]

#         cal_obs_loader = cal_obs_loaders[i]
#         cal_int_loader = cal_int_loaders[i]
#         test_int_loader = test_int_loaders[i]

#         model_u.eval()
#         model_l.eval()
        
#         # compute scores on cal_obs
#         scores_list = mf_calib(cal_obs_loader, model_u, model_l, 
#                                device=device, alpha=alpha, params=params, standardize=standardize)
        
#         plot_vec_dist(scores_list, folder_name="iDCF/figs", filename='nonconf_score_cal_obs.png')

#         D_calib_obs = get_density_ratio_data(cal_obs_loader, model_u,method, device=device)
#         D_calib_int = get_density_ratio_data(cal_int_loader, model_u,method, device=device)
#         D_test_int = get_density_ratio_data(test_int_loader, model_u,method, device=device)

#         X_calib_int = D_calib_int[:,:-1] # drop the last dimension
#         X_test_int = D_test_int[:,:-1]

#         y_calib_int =  D_calib_int[:,-1]
#         y_test_int =  D_test_int[:,-1]

#         if exact:
#             # n_calib_obs = len(D_calib_obs)
#             n_calib_int = len(D_calib_int)
#             # n_test_int = len(D_test_int)

#             n_calib_int_fold_one = n_calib_int//2

#             D_calib_int_fold_one = D_calib_int[:n_calib_int_fold_one,:]
#             X_calib_int_fold_one = X_calib_int[:n_calib_int_fold_one,:]
#             X_calib_int_fold_two = X_calib_int[n_calib_int_fold_one:,:]

#             y_calib_int_fold_one = y_calib_int[:n_calib_int_fold_one]
#             y_calib_int_fold_two = y_calib_int[n_calib_int_fold_one:]

#             if dr_model == "DR":
#                 weights_calib_obs = density_model.compute_density_ratio(D_calib_obs)
#                 weights_calib_int = density_model.compute_density_ratio(D_calib_int_fold_one)

#             elif dr_model == "MLP":
#                 p_calib_obs = density_model.predict_proba(D_calib_obs)[:,1]
#                 weights_calib_obs = (1. - p_calib_obs) / p_calib_obs #TODO: double check

#                 p_calib_int = density_model.predict_proba(D_calib_int_fold_one)[:,1]
#                 weights_calib_int = (1. - p_calib_int) / p_calib_int

#         else:
#             if dr_model == "DR":
#                 weights_calib_obs = density_model.compute_density_ratio(D_calib_obs)
#                 weights_calib_int = density_model.compute_density_ratio(D_calib_int)
            
#             elif dr_model == "MLP":
#                 p_calib_obs = density_model.predict_proba(D_calib_obs)[:,1]
#                 weights_calib_obs = (1. - p_calib_obs) / p_calib_obs #TODO: double check

#                 p_calib_int = density_model.predict_proba(D_calib_int)[:,1]
#                 weights_calib_int = (1. - p_calib_int) / p_calib_int

#         print(f"weight obs: mean {np.mean(weights_calib_obs)}, std {np.std(weights_calib_obs)}")
#         print(f"weight int: mean {np.mean(weights_calib_int)}, std {np.std(weights_calib_int)}")

#         # weights are different for exact and inexact
#         offset = weighted_conformal(alpha, weights_calib_obs, weights_calib_int, scores_list)[0]
#         print(f"offset: {offset}")

#         # compute predicted y_u, y_l on calib_int data
#         with torch.no_grad():
#             labels, y_u_list, y_l_list = list(), list(), list()
#             for index, (uid, iid, rating) in enumerate(cal_int_loader):
#                 uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
#                 y_u = model_u.predict(uid, iid)
#                 y_l = model_l.predict(uid, iid)

#                 if standardize:
#                     y_u = params["min_val"] + y_u * (params["max_val"] - params["min_val"])
#                     y_l = params["min_val"] + y_l * (params["max_val"] - params["min_val"])
                
#                 labels.extend(rating.tolist())
#                 y_u_list.append(y_u.detach().cpu().numpy())
#                 y_l_list.append(y_l.detach().cpu().numpy())
                
#             y_u_list = np.concatenate(y_u_list)
#             y_l_list = np.concatenate(y_l_list)

#             y_u_list = y_u_list + offset
#             y_l_list = y_l_list - offset

#             # inexact method: fit models to predict y_u, y_l from features

#             print("fit models for inexact upper/lower bound prediction")
#             # not quantile regression
#             y_u_predictor = RandomForestRegressor(n_estimators=n_estimators) #base_learners_dict[base_learner](**first_CQR_args_u)
#             y_l_predictor = RandomForestRegressor(n_estimators=n_estimators) #base_learners_dict[base_learner](**first_CQR_args_l)

#             if exact:
#                 y_u_fold_one = y_u_list[:n_calib_int_fold_one]
#                 y_l_fold_one = y_l_list[:n_calib_int_fold_one]
#                 # y_u_fold_two = y_u_list[n_calib_int_fold_one:]
#                 # y_l_fold_two = y_l_list[n_calib_int_fold_one:]

#                 y_u_predictor.fit(X_calib_int_fold_one, y_u_fold_one)
#                 y_l_predictor.fit(X_calib_int_fold_one, y_l_fold_one)

#                 y_u_hat_calib_int_fold_two = y_u_predictor.predict(X_calib_int_fold_two)
#                 y_l_hat_calib_int_fold_two = y_l_predictor.predict(X_calib_int_fold_two)

#                 # run conf the 2nd time, using calib_int fold one as training and fold two as calibration
#                 scores_ = np.maximum(y_l_hat_calib_int_fold_two - y_calib_int_fold_two, 
#                                y_calib_int_fold_two - y_u_hat_calib_int_fold_two)
#                 offset_ = standard_conformal(alpha, scores_)

#                 predicts_u_test_int = y_u_predictor.predict(X_test_int) + offset_
#                 predicts_l_test_int = y_l_predictor.predict(X_test_int) - offset_

#             else:

#                 y_u_predictor.fit(X_calib_int, y_u_list)
#                 y_l_predictor.fit(X_calib_int, y_l_list)

#                 print("running upper/lower bound prediction")

#                 # mse on calib data

#                 predicts_u_calib_int = y_u_predictor.predict(X_calib_int)
#                 predicts_l_calib_int = y_l_predictor.predict(X_calib_int)

#                 mse_u_calib_int = mean_squared_error(predicts_u_calib_int, y_u_list)
#                 mse_l_calib_int = mean_squared_error(predicts_l_calib_int, y_l_list)

#                 print(f"training mse of upper bound: {mse_u_calib_int}, lower bound: {mse_l_calib_int}")

#                 predicts_u_test_int = y_u_predictor.predict(X_test_int)
#                 predicts_l_test_int = y_l_predictor.predict(X_test_int)

#             # predict_u_list.append(predicts_u)
#             # predict_l_list.append(predicts_l)

#             labels = np.array(labels)
#             coverage = np.mean((labels >= predicts_l_test_int) & (labels <= predicts_u_test_int))
#             interval_width = np.mean(np.abs(predicts_u_test_int - predicts_l_test_int))

#             coverages.append(coverage)
#             interval_widths.append(interval_width)
                
#     return coverages, interval_widths

def mf_conf_eval_splitcp_mse(config, cal_obs_loaders:list, cal_int_loaders:list, test_int_loaders:list, 
                         model_list:list, dr_model_list:list, method:str,
                        device:str="cpu", params=None, alpha:float=0.1, standardize:bool=True,
                        base_learner:str="QRF", n_estimators:int=100,
                        dr_model:str="DR", dataset="coat", mix_method="concat",
                        dr_use_Y:bool=False):

    """
    split CP (exact and inexact for a single MF model trained with mse loss)
    """
    
    print("start conf eval for our split cp methods")

    # first do weighted CP on calib

    n_folds = len(cal_obs_loaders)

    coverages = []
    interval_widths = []

    seed = config["seed"]
    val_ratio = config["data_params"]["val_ratio"]
    train_ratio = config["data_params"]["train_ratio"]

    fn_str = f'_val_ratio_{val_ratio}_tr_ratio_{train_ratio}_seed_{seed}'
    
    for i in range(n_folds):
        model = model_list[i]

        density_model = dr_model_list[i]

        cal_obs_loader = cal_obs_loaders[i]
        cal_int_loader = cal_int_loaders[i]
        test_int_loader = test_int_loaders[i]

        model.eval()
        
        # normal split CP: compute scores on cal_obs
        scores_list = mf_calib_mse(cal_obs_loader, model, 
                               device=device, 
                               params=params, standardize=standardize)

        
        plot_vec_dist(scores_list, folder_name=f"iDCF/figs/{dataset}", 
                      filename=f'{method}_nonconf_score_cal_obs{fn_str}.png')

        D_calib_obs = get_density_ratio_data(cal_obs_loader, model, method, device=device, mix_method=mix_method)
        D_calib_int = get_density_ratio_data(cal_int_loader, model, method, device=device, mix_method=mix_method)
        D_test_int = get_density_ratio_data(test_int_loader, model, method, device=device, mix_method=mix_method)

        X_calib_obs = D_calib_obs[:,:-1]
        X_calib_int = D_calib_int[:,:-1] # drop the last dimension
        X_test_int = D_test_int[:,:-1]

        y_calib_int =  D_calib_int[:,-1]
        y_test_int =  D_test_int[:,-1]

        if not dr_use_Y:
            D_calib_obs = X_calib_obs
            D_calib_int = X_calib_int
            D_test_int = X_test_int

        if method == "exact":
            # half for PO, half for ITE
            alpha = alpha / 2.0

            # n_calib_obs = len(D_calib_obs)
            n_calib_int = len(D_calib_int)
            # n_test_int = len(D_test_int)

            n_calib_int_fold_one = n_calib_int//2

            D_calib_int_fold_one = D_calib_int[:n_calib_int_fold_one,:]
            X_calib_int_fold_one = X_calib_int[:n_calib_int_fold_one,:]
            X_calib_int_fold_two = X_calib_int[n_calib_int_fold_one:,:]

            y_calib_int_fold_one = y_calib_int[:n_calib_int_fold_one]
            y_calib_int_fold_two = y_calib_int[n_calib_int_fold_one:]


            if dr_model == "DR":
                weights_calib_obs = density_model.compute_density_ratio(D_calib_obs)
                weights_calib_int = density_model.compute_density_ratio(D_calib_int_fold_one)

            elif dr_model == "MLP":
                p_calib_obs = density_model.predict_proba(D_calib_obs)[:,1]
                weights_calib_obs = (1. - p_calib_obs) / p_calib_obs #TODO: double check

                p_calib_int = density_model.predict_proba(D_calib_int_fold_one)[:,1]
                weights_calib_int = (1. - p_calib_int) / p_calib_int

        elif method == "inexact":
            if dr_model == "DR":
                weights_calib_obs = density_model.compute_density_ratio(D_calib_obs)
                weights_calib_int = density_model.compute_density_ratio(D_calib_int)
            
            elif dr_model == "MLP":
                p_calib_obs = density_model.predict_proba(D_calib_obs)[:,1]
                weights_calib_obs = (1. - p_calib_obs) / p_calib_obs #TODO: double check

                p_calib_int = density_model.predict_proba(D_calib_int)[:,1]
                weights_calib_int = (1. - p_calib_int) / p_calib_int
        else:
            raise ValueError("only support [exact, inexact]")

        print(f"weight obs: mean {np.mean(weights_calib_obs)}, std {np.std(weights_calib_obs)}")
        print(f"weight int: mean {np.mean(weights_calib_int)}, std {np.std(weights_calib_int)}")

        plot_vec_dist(weights_calib_obs, 
                      folder_name=f"iDCF/figs/{dataset}", 
                      filename=f'{method}_weights_calib_obs{fn_str}.png')
        
        plot_vec_dist(weights_calib_int, 
                      folder_name=f"iDCF/figs/{dataset}", 
                      filename=f'{method}_weights_calib_int{fn_str}.png')

        # weights are different between exact and inexact
        offset = weighted_conformal(alpha, weights_calib_obs, weights_calib_int, scores_list)
        print(f"offset: {offset}")

        # compute predicted y_u, y_l on calib_int data
        with torch.no_grad():
            labels, predicts = list(), list()
            for index, (uid, iid, rating) in enumerate(cal_int_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                y = model.predict(uid, iid)

                if standardize:
                    y = params["min_val"] + y * (params["max_val"] - params["min_val"])
                
                labels.extend(rating.tolist())
                predicts.append(y.detach().cpu().numpy())
                
            predicts = np.concatenate(predicts)

            plot_vec_dist(predicts, folder_name=f"iDCF/figs/{dataset}",
                           filename=f'{method}_predicts_cal_int{fn_str}.png', offset=offset)
            try:
                labels = np.concatenate(labels)
            except:
                labels = np.array(labels)
            scores_list = np.abs(predicts-labels)
            plot_vec_dist(scores_list, folder_name=f"iDCF/figs/{dataset}",
                           filename=f'{method}_nonconf_score_test_int{fn_str}.png', offset=offset)

            if method == "exact":
                # for exact method
                # the offset is only for fold one
                y_u_list = predicts[:n_calib_int_fold_one] + offset
                y_l_list = predicts[:n_calib_int_fold_one] - offset

            elif method == "inexact":
                y_u_list = predicts + offset
                y_l_list = predicts - offset
            
            else:
                raise ValueError("method not implemented")

            # inexact method: fit models to predict y_u, y_l from features

            print("fit models for inexact upper/lower bound prediction")
            
            n_estimators_target = n_estimators
            if base_learner == "GBM":
                second_CQR_args_u = dict({"loss": "quantile", "alpha":0.55, "n_estimators": n_estimators_target})
                second_CQR_args_l = dict({"loss": "quantile", "alpha":0.45, "n_estimators": n_estimators_target})
                y_u_predictor = base_learners_dict[base_learner](**second_CQR_args_u)
                y_l_predictor = base_learners_dict[base_learner](**second_CQR_args_l)

            elif base_learner == "QRF":
                second_CQR_args_u = dict({"default_quantiles":0.6, "n_estimators": n_estimators_target}) 
                second_CQR_args_l = dict({"default_quantiles":0.4, "n_estimators": n_estimators_target})
                y_u_predictor = base_learners_dict[base_learner](**second_CQR_args_u)
                y_l_predictor = base_learners_dict[base_learner](**second_CQR_args_l)

            elif base_learner == "RF":
                y_u_predictor = RandomForestRegressor(n_estimators=n_estimators) 
                y_l_predictor = RandomForestRegressor(n_estimators=n_estimators)

            if method == "exact":
                y_u_fold_one = y_u_list[:n_calib_int_fold_one]
                y_l_fold_one = y_l_list[:n_calib_int_fold_one]
                # y_u_fold_two = y_u_list[n_calib_int_fold_one:]
                # y_l_fold_two = y_l_list[n_calib_int_fold_one:]

                y_u_predictor.fit(X_calib_int_fold_one, y_u_fold_one)
                y_l_predictor.fit(X_calib_int_fold_one, y_l_fold_one)

                y_u_hat_calib_int_fold_two = y_u_predictor.predict(X_calib_int_fold_two)
                y_l_hat_calib_int_fold_two = y_l_predictor.predict(X_calib_int_fold_two)

                # run conf the 2nd time, using calib_int fold one as training and fold two as calibration
                print("second calibration for exact method...")
                scores_ = np.maximum(y_l_hat_calib_int_fold_two - y_calib_int_fold_two, 
                               y_calib_int_fold_two - y_u_hat_calib_int_fold_two)
                offset_ = standard_conformal(alpha, scores_)

                predicts_u_test_int = y_u_predictor.predict(X_test_int) + offset_
                predicts_l_test_int = y_l_predictor.predict(X_test_int) - offset_

            else:

                y_u_predictor.fit(X_calib_int, y_u_list)
                y_l_predictor.fit(X_calib_int, y_l_list)

                print("running upper/lower bound prediction")

                # mse on calib data

                predicts_u_calib_int = y_u_predictor.predict(X_calib_int)
                predicts_l_calib_int = y_l_predictor.predict(X_calib_int)

                mse_u_calib_int = mean_squared_error(predicts_u_calib_int, y_u_list)
                mse_l_calib_int = mean_squared_error(predicts_l_calib_int, y_l_list)

                print(f"training mse of upper bound: {mse_u_calib_int}, lower bound: {mse_l_calib_int}")

                predicts_u_test_int = y_u_predictor.predict(X_test_int)
                predicts_l_test_int = y_l_predictor.predict(X_test_int)

            # predict_u_list.append(predicts_u)
            # predict_l_list.append(predicts_l)

            # labels = np.array(labels)
            y_test_int = np.array(y_test_int)
            coverage = np.mean((y_test_int >= predicts_l_test_int) & (y_test_int <= predicts_u_test_int))
            interval_width = np.mean(np.abs(predicts_u_test_int - predicts_l_test_int))

            coverages.append(coverage)
            interval_widths.append(interval_width)
                
    return coverages, interval_widths

# def mf_conf_eval_naive(cal_loaders:list, test_loaders:list, model_u_list:list, model_l_list:list, 
#                  device="cpu", params=None, alpha=0.1, standardize=True):

#     n_folds = len(cal_loaders)

#     # offset_list = []
#     # predict_u_list = []
#     # predict_l_list = []

#     coverages = []
#     interval_widths = []

#     for i in range(n_folds):
#         model_u = model_u_list[i]
#         model_l = model_l_list[i]
#         cal_loader = cal_loaders[i]
#         test_loader = test_loaders[i]

#         model_u.eval()
#         model_l.eval()
        
#         scores_list = mf_calib(cal_loader, model_u, model_l, device=device, alpha=alpha, standardize=standardize)
#         plot_vec_dist(scores_list, folder_name="iDCF/figs", filename='nonconf_score_cal_int.png')

#         offset = standard_conformal(alpha, scores_list)

#         with torch.no_grad():
#             labels, predicts_u, predicts_l = list(), list(), list()
#             for index, (uid, iid, rating) in enumerate(test_loader):
#                 uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
#                 predict_u = model_u.predict(uid, iid)
#                 predict_l = model_l.predict(uid, iid)
                
#                 if standardize:
#                     predict_u = params["min_val"] + predict_u * (params["max_val"] - params["min_val"])
#                     predict_l = params["min_val"] + predict_l * (params["max_val"] - params["min_val"])
                
#                 labels.extend(rating.tolist())
#                 predicts_u.append(predict_u.detach().cpu().numpy())
#                 predicts_l.append(predict_l.detach().cpu().numpy())
                
#             predicts_u = np.concatenate(predicts_u)
#             predicts_l = np.concatenate(predicts_l)

#             predicts_u = predicts_u + offset
#             predicts_l = predicts_l - offset

#             # predict_u_list.append(predicts_u)
#             # predict_l_list.append(predicts_l)

#             labels = np.array(labels)
#             coverage = np.mean((labels >= predicts_l) & (labels <= predicts_u))
#             interval_width = np.mean(np.abs(predicts_u - predicts_l))

#             coverages.append(coverage)
#             interval_widths.append(interval_width)
                
#     return coverages, interval_widths

def mf_conf_eval_naive_mse(config, cal_int_loaders:list, test_int_loaders:list, model_list:list, method:str,
                 device="cpu", params=None, alpha=0.1, standardize=True, dataset="coat", truncate:bool=False):

    n_folds = len(cal_int_loaders)

    # offset_list = []
    # predict_u_list = []
    # predict_l_list = []

    coverages = []
    interval_widths = []

    seed = config["seed"]
    val_ratio = config["data_params"]["val_ratio"]
    train_ratio = config["data_params"]["train_ratio"]

    fn_str = f'_val_ratio_{val_ratio}_tr_ratio_{train_ratio}_seed_{seed}'

    for i in range(n_folds):
        model = model_list[i]
        cal_int_loader = cal_int_loaders[i]
        test_int_loader = test_int_loaders[i]

        model.eval()
        
        scores_list = mf_calib_mse(cal_int_loader, model, device=device, standardize=standardize, params=params)
        plot_vec_dist(scores_list, folder_name=f"iDCF/figs/{dataset}", filename=f'{method}_nonconf_score_cal_int{fn_str}.png')

        offset = standard_conformal(alpha, scores_list)

        with torch.no_grad():
            labels, predicts = list(), list()
            for index, (uid, iid, rating) in enumerate(test_int_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict = model.predict(uid, iid)
                
                if standardize:
                    predict = params["min_val"] + predict * (params["max_val"] - params["min_val"])
                
                labels.extend(rating.tolist())
                predicts.append(predict.detach().cpu().numpy())
                
            # predicts_u = np.concatenate(predicts_u)
            predicts = np.concatenate(predicts)
            try:
                labels = np.concatenate(labels)
            except:
                labels = np.array(labels)

            scores_list = np.abs(predicts - labels)
            plot_vec_dist(scores_list, folder_name=f"iDCF/figs/{dataset}",
                           filename=f'{method}_nonconf_score_test_int{fn_str}.png', offset=offset)

            plot_vec_dist(predicts, folder_name=f"iDCF/figs/{dataset}",
                           filename=f'{method}_predicts_test_int{fn_str}.png', offset=offset)

            predicts_u = predicts + offset
            predicts_l = predicts - offset

            # predict_u_list.append(predicts_u)
            # predict_l_list.append(predicts_l)

            labels = np.array(labels)
            coverage = np.mean((labels >= predicts_l) & (labels <= predicts_u))
            interval_width = np.mean(np.abs(predicts_u - predicts_l))

            coverages.append(coverage)
            interval_widths.append(interval_width)
                
    return coverages, interval_widths


def mf_conf_eval_wcp_mse(config, cal_obs_loaders:list, cal_int_loaders:list, test_int_loaders:list,
                          model_list:list, dr_model_list:list, 
                          method:str,
                        device="cpu", params=None, alpha=0.1, standardize=True, 
                        dataset="coat", dr_model:str="DR", mix_method="concat"):

    # cal_loaders: calib obs
    # test_loaders: test int

    # def weight_1(model, x):
    #     pscores = model.predict_proba(x)[:, 1]
    #     return 1. / pscores

    n_folds = len(cal_obs_loaders)

    # offset_list = []
    # predict_u_list = []
    # predict_l_list = []

    coverages = []
    interval_widths = []


    seed = config["seed"]
    val_ratio = config["data_params"]["val_ratio"]
    train_ratio = config["data_params"]["train_ratio"]

    fn_str = f'_val_ratio_{val_ratio}_tr_ratio_{train_ratio}_seed_{seed}'

    for i in range(n_folds):
        model = model_list[i]
        cal_obs_loader = cal_obs_loaders[i]
        cal_int_loader = cal_int_loaders[i]
        test_int_loader = test_int_loaders[i]

        density_model = dr_model_list[i]

        model.eval()
        
        scores_list = mf_calib_mse(cal_obs_loader, model, 
                                   device=device,
                                   standardize=standardize, 
                                   params=params)
        
        plot_vec_dist(scores_list, 
                      folder_name=f"iDCF/figs/{dataset}", 
                      filename=f'{method}_nonconf_score_cal_obs{fn_str}.png')
        
        if method == "wcp":
            raise ValueError("this method does not make sense")
            # D is just embeddings for wcp
            # X_calib_obs = get_density_ratio_data(cal_obs_loader, model, method, device=device, mix_method=mix_method)
            # X_calib_int = get_density_ratio_data(cal_int_loader, model, method, device=device, mix_method=mix_method)

            # if dr_model == "DR":
            #     weights_calib_obs = density_model.compute_density_ratio(X_calib_obs)
            #     weights_calib_int = density_model.compute_density_ratio(X_calib_int)

            # elif dr_model == "MLP":
            #     p_calib_obs = density_model.predict_proba(X_calib_obs)[:,1]
            #     weights_calib_obs = (1. - p_calib_obs) / p_calib_obs #TODO: double check

            #     p_calib_int = density_model.predict_proba(X_calib_int)[:,1]
            #     weights_calib_int = (1. - p_calib_int) / p_calib_int
        
        
        elif method == "wcp_ips":
            weights_calib_obs = get_ips_weights(cal_obs_loader, model, method, device="cpu")
            weights_test_int = get_ips_weights(test_int_loader, model, method, device="cpu")

        else:
            raise ValueError("method not supported")

        # should the offset be different for each test sample?

        plot_vec_dist(weights_calib_obs, 
                      folder_name=f"iDCF/figs/{dataset}", 
                      filename=f'{method}_weights_calib_obs{fn_str}.png')
        
        plot_vec_dist(weights_test_int, 
                      folder_name=f"iDCF/figs/{dataset}", 
                      filename=f'{method}_weights_test_int{fn_str}.png')

        offset = weighted_conformal(alpha, 
                                    weights_calib_obs, 
                                    weights_test_int, scores_list)

        with torch.no_grad():
            labels, predicts = list(), list()
            for index, (uid, iid, rating) in enumerate(test_int_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict = model.predict(uid, iid)
                
                if standardize:
                    predict = params["min_val"] + predict * (params["max_val"] - params["min_val"])
                
                labels.extend(rating.tolist())
                predicts.append(predict.detach().cpu().numpy())
                
            # predicts_u = np.concatenate(predicts_u)
            predicts = np.concatenate(predicts)
            plot_vec_dist(predicts, folder_name=f"iDCF/figs/{dataset}",
                           filename=f'{method}_predicts_test_int{fn_str}.png', offset=offset)
            
            try:
                labels = np.concatenate(labels)
            except:
                labels = np.array(labels)

            scores_list = np.abs(predicts-labels)
            plot_vec_dist(scores_list, folder_name=f"iDCF/figs/{dataset}",
                           filename=f'{method}_nonconf_score_test_int{fn_str}.png', offset=offset)

            predicts_u = predicts + offset
            predicts_l = predicts - offset

            # predict_u_list.append(predicts_u)
            # predict_l_list.append(predicts_l)

            labels = np.array(labels)
            coverage = np.mean(
                (labels >= predicts_l) & (labels <= predicts_u))
            interval_width = np.mean(np.abs(predicts_u - predicts_l))

            coverages.append(coverage)
            interval_widths.append(interval_width)
                
    return coverages, interval_widths