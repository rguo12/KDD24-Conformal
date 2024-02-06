from __future__ import absolute_import, division, print_function

import sys, os, time, random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from quantile_forest import RandomForestQuantileRegressor
from sklearn import preprocessing
from sklearn.base import clone
from joblib import Parallel, delayed
# import density_ratio_estimation.src.densityratio as densityratio
from densratio import densratio
# from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from functools import partial
import models.utils as utils

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Global options for baselearners (see class attributes below)

base_learners_dict = dict({"GBM": GradientBoostingRegressor, 
                           "RF": RandomForestQuantileRegressor})

class BaseCP:
    
    def __init__(self, data_obs, data_inter, n_folds,
                 alpha=0.1, base_learner="RF", 
                 quantile_regression=True, n_estimators : int = 10):

        """
        Base class for conformal prediction, including transductive and split naive, inexact and exact.

            :param n_folds: the number of folds for the DR learner cross-fitting (See [1])
            :param alpha: the target miscoverage level. alpha=.1 means that target coverage is 90%
            :param base_learner: the underlying regression model
                                - current options: ["GBM": gradient boosting machines, "RF": random forest]
            :param quantile_regression: Boolean for indicating whether the base learner is a quantile regression model
                                        or a point estimate of the CATE function. 

        """

        # set base learner
        self.base_learner = base_learner
        self.quantile_regression = quantile_regression
        self.n_folds = n_folds
        self.alpha = alpha
        # n_estimators_nuisance = 100
        # n_estimators_target = 100
        
        if self.quantile_regression:
            if self.base_learner == "GBM":
                self.first_CQR_args_u = dict({"loss": "quantile", "alpha":1 - (self.alpha / 2), "n_estimators": n_estimators}) 
                self.first_CQR_args_l = dict({"loss": "quantile", "alpha":self.alpha/2, "n_estimators": n_estimators})

            elif self.base_learner == "RF":
                self.first_CQR_args_u = dict({"default_quantiles":1 - (self.alpha/2), "n_estimators": n_estimators})
                self.first_CQR_args_l = dict({"default_quantiles":self.alpha/2, "n_estimators": n_estimators})
            else:
                raise ValueError('base_learner must be one of GBM or RF')
            
        else:
            if self.base_learner == "GBM":
                self.first_CQR_args = dict({"loss": "squared_error", "n_estimators": n_estimators}) 
            elif self.base_learner == "RF":
                self.first_CQR_args = dict({"criterion": "squared_error", "n_estimators": n_estimators}) 
            else:
                raise ValueError('base_learner must be one of GBM or RF')

        
        # pseudo label model
        if self.base_learner == "GBM":
            self.pseudo_label_args = dict({"loss": "squared_error", "n_estimators": n_estimators}) 
        elif self.base_learner == "RF":
            self.pseudo_label_args = dict({"criterion": "squared_error", "n_estimators": n_estimators}) 

        self.data_obs = data_obs
        self.data_inter = data_inter
        self.train_obs_index_list, self.X_train_obs_list, self.T_train_obs_list, self.Y_train_obs_list, self.calib_obs_index_list, self.X_calib_obs_list, self.T_calib_obs_list, self.Y_calib_obs_list = utils.split_data(self.data_obs, n_folds)
        self.train_inter_index_list, self.X_train_inter_list, self.T_train_inter_list, self.Y_train_inter_list, self.calib_inter_index_list, self.X_calib_inter_list, self.T_calib_inter_list, self.Y_calib_inter_list = utils.split_data(self.data_inter, n_folds)      
        
        n_estimators_target = 100

        # if self.base_learner == "GBM":
        #     second_CQR_args_u = dict({"loss": "quantile", "alpha":0.6, "n_estimators": n_estimators_target})
        #     second_CQR_args_l = dict({"loss": "quantile", "alpha":0.4, "n_estimators": n_estimators_target})
        # elif self.base_learner == "RF":
        #     second_CQR_args_u = dict({"default_quantiles":0.6, "n_estimators": n_estimators_target}) 
        #     second_CQR_args_l = dict({"default_quantiles":0.4, "n_estimators": n_estimators_target})
        
        # self.tilde_C_ITE_model_u = [base_learners_dict[self.base_learner](**second_CQR_args_u) for _ in range(self.n_folds)] 
        # self.tilde_C_ITE_model_l = [base_learners_dict[self.base_learner](**second_CQR_args_l) for _ in range(self.n_folds)] 

        return

    def fit(self, method):
        # Implement the common fit logic here
        pass


    def reset_tilde_C_ITE_models(self, cf_method:str, n_estimators:int=100):
        # for ITE estimation
        # using MSE loss
        
        # if self.base_learner == "GBM":
        #     second_CQR_args_u = dict({"loss": "quantile", "alpha":0.6, "n_estimators": n_estimators_target})
        #     second_CQR_args_l = dict({"loss": "quantile", "alpha":0.4, "n_estimators": n_estimators_target})
        # elif self.base_learner == "RF":
        #     second_CQR_args_u = dict({"default_quantiles":0.6, "n_estimators": n_estimators_target}) 
        #     second_CQR_args_l = dict({"default_quantiles":0.4, "n_estimators": n_estimators_target})

        if self.base_learner == "GBM":
            first_CQR_args = dict({"loss": "squared_error", "n_estimators": n_estimators}) 
        elif self.base_learner == "RF":
            first_CQR_args = dict({"criterion": "squared_error", "n_estimators": n_estimators}) 
        else:
            raise ValueError('base_learner must be one of GBM or RF')
        
        if cf_method == "naive":
            self.tilde_C_ITE_model_u = [base_learners_dict[self.base_learner](**first_CQR_args) for _ in range(self.n_folds)] 
            self.tilde_C_ITE_model_l = [base_learners_dict[self.base_learner](**first_CQR_args) for _ in range(self.n_folds)] 
        elif cf_method in ["exact", "inexact"]:
            self.tilde_C_ITE_model_u = [[base_learners_dict[self.base_learner](**first_CQR_args) for _ in range(self.n_folds)]for _ in range(self.n_folds)] 
            self.tilde_C_ITE_model_l = [[base_learners_dict[self.base_learner](**first_CQR_args) for _ in range(self.n_folds)]for _ in range(self.n_folds)] 
        

    # def predict_counterfactual_inexact(self, alpha, X_test, Y0, Y1):
    #     # Implement the common predict_counterfactual_inexact logic here
    #     pass

    # def predict_counterfactual_exact(self, alpha, X_test, Y0, Y1):
    #     # Implement the common predict_counterfactual_exact logic here
    #     pass

    # def predict_counterfactual_naive(self, alpha, X_test, Y0, Y1):
    #     # Implement the common predict_counterfactual_naive logic here
    #     pass


class SplitCP(BaseCP):

    def __init__(self, data_obs, data_inter, n_folds,
                 alpha=0.1, base_learner="GBM", quantile_regression=True):

        """
        Split conformal prediction, including naive, inexact and exact.

            :param n_folds: the number of folds for the DR learner cross-fitting (See [1])
            :param alpha: the target miscoverage level. alpha=.1 means that target coverage is 90%
            :param base_learner: the underlying regression model
                                - current options: ["GBM": gradient boosting machines, "RF": random forest]
            :param quantile_regression: Boolean for indicating whether the base learner is a quantile regression model
                                        or a point estimate of the CATE function. 

        """
        super().__init__(data_obs, data_inter, n_folds, alpha, base_learner, quantile_regression)
        self.fitted = False



    def fit(self, method, dr_use_Y=1):
        """
        Fits the plug-in models and meta-learners using the sample (X, W, Y) and true propensity scores pscores
        """
        if method in ['two_stage_inexact', 'two_stage_exact']:
            self.models_u_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_l_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_u_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_l_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)] 

            self.density_models_0 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.density_models_1 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]

            self.models_pseudo_label_0 = [[base_learners_dict[self.base_learner](**self.pseudo_label_args) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_pseudo_label_1 = [[base_learners_dict[self.base_learner](**self.pseudo_label_args) for _ in range(self.n_folds)] for _ in range(self.n_folds)]

            self.C0_l_model = RandomForestRegressor()
            self.C0_u_model = RandomForestRegressor()
            self.C1_l_model = RandomForestRegressor()
            self.C1_u_model = RandomForestRegressor()

            # loop over the cross-fitting folds
            for j in range(self.n_folds):
                X_train_inter_0 = self.X_train_inter_list[j][self.T_train_inter_list[j]==0, :]
                Y_train_inter_0 = self.Y_train_inter_list[j][self.T_train_inter_list[j]==0]
                X_train_inter_1 = self.X_train_inter_list[j][self.T_train_inter_list[j]==1, :]
                Y_train_inter_1 = self.Y_train_inter_list[j][self.T_train_inter_list[j]==1]

                for i in tqdm(range(self.n_folds)):
                    # i == j is okay as obs and inter data are different

                    X_train_obs_0 = self.X_train_obs_list[i][self.T_train_obs_list[i]==0, :]
                    Y_train_obs_0 = self.Y_train_obs_list[i][self.T_train_obs_list[i]==0]
                    X_train_obs_1 = self.X_train_obs_list[i][self.T_train_obs_list[i]==1, :]
                    Y_train_obs_1 = self.Y_train_obs_list[i][self.T_train_obs_list[i]==1]

                    self.models_u_0[j][i].fit(X_train_obs_0, Y_train_obs_0)
                    self.models_l_0[j][i].fit(X_train_obs_0, Y_train_obs_0)
                    self.models_u_1[j][i].fit(X_train_obs_1, Y_train_obs_1)
                    self.models_l_1[j][i].fit(X_train_obs_1, Y_train_obs_1)
                    
                    self.models_pseudo_label_0[j][i].fit(X_train_obs_0, Y_train_obs_0)
                    self.models_pseudo_label_1[j][i].fit(X_train_obs_1, Y_train_obs_1)
                    
                    D_train_obs_0, D_train_inter_0 = utils.get_dr_data(
                        X_train_obs_0, Y_train_obs_0, X_train_inter_0, Y_train_inter_0, dr_use_Y, self.models_pseudo_label_0[j][i], train=True)
                    
                    D_train_obs_1, D_train_inter_1 = utils.get_dr_data(
                        X_train_obs_1, Y_train_obs_1, X_train_inter_1, Y_train_inter_1, dr_use_Y, self.models_pseudo_label_1[j][i], train=True)
                    
                    self.density_models_0[j][i] = densratio(D_train_inter_0, D_train_obs_0, verbose=False, alpha=0.01)
                    self.density_models_1[j][i] = densratio(D_train_inter_1, D_train_obs_1, verbose=False, alpha=0.01)
        
        # elif method == 'wtcpdr':
        #     self.models_u_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
        #     self.models_l_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
        #     self.models_u_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
        #     self.models_l_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)] 

        #     self.density_models_0 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]
        #     self.density_models_1 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]
                
        #     X_inter_0 = self.X_inter[self.T_inter==0, :]
        #     Y_inter_0 = self.Y_inter[self.T_inter==0]
        #     X_inter_1 = self.X_inter[self.T_inter==1, :]
        #     Y_inter_1 = self.Y_inter[self.T_inter==1]

        #     X_obs_0 = self.X_obs[self.T_obs==0, :]
        #     Y_obs_0 = self.Y_obs[self.T_obs==0]
        #     X_obs_1 = self.X_obs[self.T_obs==1, :]
        #     Y_obs_1 = self.Y_obs[self.T_obs==1]

        #     X_aug_0, Y_aug_0 = np.concatenate((X_aug_0, x), axis=0), np.concatenate((Y_aug_0, y), axis=0)
        #     X_aug_1, Y_aug_1 = np.concatenate((X_aug_1, x), axis=0), np.concatenate((Y_aug_1, y), axis=0)

        #     self.models_u_0.fit(X_aug_0, Y_aug_0)
        #     self.models_l_0.fit(X_aug_0, Y_aug_0)
        #     self.models_u_1.fit(X_aug_1, Y_aug_1)
        #     self.models_l_1.fit(X_aug_1, Y_aug_1)
            
        #     D_obs_0 = np.concatenate((X_obs_0, Y_obs_0[:, None]), axis=1)
        #     D_inter_0 = np.concatenate((X_inter_0, Y_inter_0[:, None]), axis=1)
        #     D_obs_1 = np.concatenate((X_obs_1, Y_obs_1[:, None]), axis=1)
        #     D_inter_1 = np.concatenate((X_inter_1, Y_inter_1[:, None]), axis=1)
            
        #     self.density_models_0[j][i] = densratio(D_inter_0, D_obs_0, verbose=False, alpha=0.01)
        #     self.density_models_1[j][i] = densratio(D_inter_1, D_obs_1, verbose=False, alpha=0.01)

        elif method == 'naive':
            self.models_u_0 = [base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)]
            self.models_l_0 = [base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)]
            self.models_u_1 = [base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)]
            self.models_l_1 = [base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)]

            for j in range(self.n_folds):
                # trained by one split of the inter data
                X_train_inter_0 = self.X_train_inter_list[j][self.T_train_inter_list[j]==0, :]
                Y_train_inter_0 = self.Y_train_inter_list[j][self.T_train_inter_list[j]==0]
                X_train_inter_1 = self.X_train_inter_list[j][self.T_train_inter_list[j]==1, :]
                Y_train_inter_1 = self.Y_train_inter_list[j][self.T_train_inter_list[j]==1]

                self.models_u_0[j].fit(X_train_inter_0, Y_train_inter_0)
                self.models_l_0[j].fit(X_train_inter_0, Y_train_inter_0)
                self.models_u_1[j].fit(X_train_inter_1, Y_train_inter_1)
                self.models_l_1[j].fit(X_train_inter_1, Y_train_inter_1)
        
        self.fitted=True

    def predict_counterfactual_inexact(self, alpha, X_test, Y0, Y1, dr_model = "DR", dr_use_Y:int = 1):
        print("Fitting models ... ")
        self.fit(method='two_stage_inexact', dr_use_Y=dr_use_Y)
        print("Fitting models done. ")
        
        C_calib_u_0, C_calib_l_0 = [], []
        C_calib_u_1, C_calib_l_1 = [], []
        X_calib_inter_0_all, X_calib_inter_1_all = [], []

        for j in range(self.n_folds):
            X_calib_inter_0 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==0, :]
            Y_calib_inter_0 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==0]
            X_calib_inter_1 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==1, :]
            Y_calib_inter_1 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==1]

            offset_0_list, offset_1_list = [] , []
            y0_l_list, y0_u_list = [], []
            y1_l_list, y1_u_list = [], []

            for i in range(self.n_folds):
                X_calib_obs_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
                Y_calib_obs_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
                X_calib_obs_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
                Y_calib_obs_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

                # if dr_use_Y == 1:
                #     D_calib_obs_0 = np.concatenate((X_calib_obs_0, Y_calib_obs_0[:, None]), axis=1)
                #     D_calib_inter_0 = np.concatenate((X_calib_inter_0, Y_calib_inter_0[:, None]), axis=1)
                #     D_calib_obs_1 = np.concatenate((X_calib_obs_1, Y_calib_obs_1[:, None]), axis=1)
                #     D_calib_inter_1 = np.concatenate((X_calib_inter_1, Y_calib_inter_1[:, None]), axis=1)
                # elif dr_use_Y == 0:
                #     D_calib_obs_0 = X_calib_obs_0
                #     D_calib_inter_0 = X_calib_inter_0
                #     D_calib_obs_1 = X_calib_obs_1
                #     D_calib_inter_1 = X_calib_inter_1

                D_calib_obs_0, D_calib_inter_0 = utils.get_dr_data(
                    X_calib_obs_0, Y_calib_obs_0, X_calib_inter_0, Y_calib_inter_0, 
                    dr_use_Y, self.models_pseudo_label_0[j][i], train=False)
            
                D_calib_obs_1, D_calib_inter_1 = utils.get_dr_data(
                    X_calib_obs_1, Y_calib_obs_1, X_calib_inter_1, Y_calib_inter_1, 
                    dr_use_Y, self.models_pseudo_label_1[j][i], train=False)

                weights_calib_obs_0 = self.density_models_0[j][i].compute_density_ratio(D_calib_obs_0)
                weights_calib_inter_0 = self.density_models_0[j][i].compute_density_ratio(D_calib_inter_0)
                weights_calib_obs_1 = self.density_models_1[j][i].compute_density_ratio(D_calib_obs_1)
                weights_calib_inter_1 = self.density_models_1[j][i].compute_density_ratio(D_calib_inter_1)
            
                scores_0 = np.maximum(self.models_l_0[j][i].predict(X_calib_obs_0) - Y_calib_obs_0, Y_calib_obs_0 - self.models_u_0[j][i].predict(X_calib_obs_0))
                offset_0 = utils.weighted_conformal(alpha, weights_calib_obs_0, weights_calib_inter_0, scores_0)
                offset_0_list.append(offset_0)

                scores_1 = np.maximum(self.models_l_1[j][i].predict(X_calib_obs_1) - Y_calib_obs_1, Y_calib_obs_1 - self.models_u_1[j][i].predict(X_calib_obs_1))
                offset_1 = utils.weighted_conformal(alpha, weights_calib_obs_1, weights_calib_inter_1, scores_1)
                offset_1_list.append(offset_1)

                y0_l = self.models_l_0[j][i].predict(X_calib_inter_0)
                y0_u = self.models_u_0[j][i].predict(X_calib_inter_0)
                y0_l_list.append(y0_l)
                y0_u_list.append(y0_u)

                y1_l = self.models_l_1[j][i].predict(X_calib_inter_1)
                y1_u = self.models_u_1[j][i].predict(X_calib_inter_1)
                y1_l_list.append(y1_l)
                y1_u_list.append(y1_u)

            y0_l = np.median(np.array(y0_l_list), axis=0) - np.median(np.array(offset_0_list), axis=0)
            y0_u = np.median(np.array(y0_u_list), axis=0) + np.median(np.array(offset_0_list), axis=0)
            y1_l = np.median(np.array(y1_l_list), axis=0) - np.median(np.array(offset_1_list), axis=0)
            y1_u = np.median(np.array(y1_u_list), axis=0) + np.median(np.array(offset_1_list), axis=0)

            C_calib_u_0.append(y0_u)
            C_calib_u_1.append(y1_u)
            C_calib_l_0.append(y0_l)
            C_calib_l_1.append(y1_l)

            X_calib_inter_0_all.append(X_calib_inter_0)
            X_calib_inter_1_all.append(X_calib_inter_1)
        
        X_calib_inter_1_all = np.concatenate(X_calib_inter_1_all, axis=0)
        X_calib_inter_0_all = np.concatenate(X_calib_inter_0_all, axis=0)
        C_calib_u_0 = np.concatenate(C_calib_u_0, axis=0)
        C_calib_u_1 = np.concatenate(C_calib_u_1, axis=0)
        C_calib_l_0 = np.concatenate(C_calib_l_0, axis=0)
        C_calib_l_1 = np.concatenate(C_calib_l_1, axis=0)

        self.C0_l_model.fit(X_calib_inter_0_all, C_calib_l_0)
        self.C0_u_model.fit(X_calib_inter_0_all, C_calib_u_0)
        self.C1_l_model.fit(X_calib_inter_1_all, C_calib_l_1)
        self.C1_u_model.fit(X_calib_inter_1_all, C_calib_u_1)

        # C0_test_l = self.C0_l_model.predict(X_test)
        # C0_test_u = self.C0_u_model.predict(X_test)
        # C1_test_l = self.C1_l_model.predict(X_test)
        # C1_test_u = self.C1_u_model.predict(X_test)

        return self.C0_l_model, self.C0_u_model, self.C1_l_model, self.C1_u_model
    
    def predict_counterfactual_exact(self, alpha, X_test, Y0, Y1, dr_use_Y:bool=True):
        print("Fitting models ... ")
        self.fit(method='two_stage_exact', dr_use_Y=dr_use_Y)
        print("Fitting models done. ")
        
        C_calib_u_0, C_calib_l_0 = [], []
        C_calib_u_1, C_calib_l_1 = [], []
        X_calib_inter_0_fold_one_list, X_calib_inter_1_fold_one_list = [], []
        X_calib_inter_0_fold_two_list, X_calib_inter_1_fold_two_list = [], []
        Y_calib_inter_0_fold_two_list, Y_calib_inter_1_fold_two_list = [], []

        for j in range(self.n_folds):
            X_calib_inter_0 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==0, :]
            Y_calib_inter_0 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==0]
            X_calib_inter_1 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==1, :]
            Y_calib_inter_1 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==1]

            # split calib inter data into two folds
            calib_num_0, calib_num_1 = len(X_calib_inter_0), len(X_calib_inter_1)
            n_fold_one_0 = int(calib_num_0/2)
            n_fold_one_1 = int(calib_num_1/2)
            X_calib_inter_0_fold_one, X_calib_inter_0_fold_two = X_calib_inter_0[:n_fold_one_0, :], X_calib_inter_0[n_fold_one_0:, :]
            Y_calib_inter_0_fold_one, Y_calib_inter_0_fold_two = Y_calib_inter_0[:n_fold_one_0], Y_calib_inter_0[n_fold_one_0:]
            X_calib_inter_1_fold_one, X_calib_inter_1_fold_two = X_calib_inter_1[:n_fold_one_1, :], X_calib_inter_1[n_fold_one_1:, :]
            Y_calib_inter_1_fold_one, Y_calib_inter_1_fold_two = Y_calib_inter_1[:n_fold_one_1], Y_calib_inter_1[n_fold_one_1:]

            offset_0_list, offset_1_list = [] , []
            y0_l_list, y0_u_list = [], []
            y1_l_list, y1_u_list = [], []

            for i in range(self.n_folds):
                X_calib_obs_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
                Y_calib_obs_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
                X_calib_obs_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
                Y_calib_obs_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

                # only use one fold of calib_int data to compute weights dr model
                # if dr_use_Y:
                #     D_calib_obs_0 = np.concatenate((X_calib_obs_0, Y_calib_obs_0[:, None]), axis=1)
                #     D_calib_inter_0 = np.concatenate((X_calib_inter_0, Y_calib_inter_0[:, None]), axis=1)
                #     D_calib_obs_1 = np.concatenate((X_calib_obs_1, Y_calib_obs_1[:, None]), axis=1)
                #     D_calib_inter_1 = np.concatenate((X_calib_inter_1, Y_calib_inter_1[:, None]), axis=1)
                # else:
                #     D_calib_obs_0 = X_calib_obs_0
                #     D_calib_inter_0 = X_calib_inter_0
                #     D_calib_obs_1 = X_calib_obs_1
                #     D_calib_inter_1 = X_calib_inter_1

                D_calib_obs_0, D_calib_inter_0 = utils.get_dr_data(
                    X_calib_obs_0, Y_calib_obs_0, X_calib_inter_0, Y_calib_inter_0, dr_use_Y, self.models_pseudo_label_0[j][i], train=False)
            
                D_calib_obs_1, D_calib_inter_1 = utils.get_dr_data(
                    X_calib_obs_1, Y_calib_obs_1, X_calib_inter_1, Y_calib_inter_1, dr_use_Y, self.models_pseudo_label_1[j][i], train=False)

                weights_calib_obs_0 = self.density_models_0[j][i].compute_density_ratio(D_calib_obs_0)
                weights_calib_inter_0 = self.density_models_0[j][i].compute_density_ratio(D_calib_inter_0)
                weights_calib_obs_1 = self.density_models_1[j][i].compute_density_ratio(D_calib_obs_1)
                weights_calib_inter_1 = self.density_models_1[j][i].compute_density_ratio(D_calib_inter_1)

                # still use calib_obs to compute nonconf scores
                scores_0 = np.maximum(self.models_l_0[j][i].predict(X_calib_obs_0) - Y_calib_obs_0,
                                       Y_calib_obs_0 - self.models_u_0[j][i].predict(X_calib_obs_0))
                offset_0 = utils.weighted_conformal(alpha, weights_calib_obs_0, weights_calib_inter_0, scores_0)
                offset_0_list.append(offset_0[:n_fold_one_0])

                scores_1 = np.maximum(self.models_l_1[j][i].predict(X_calib_obs_1) - Y_calib_obs_1,
                                       Y_calib_obs_1 - self.models_u_1[j][i].predict(X_calib_obs_1))
                offset_1 = utils.weighted_conformal(alpha, weights_calib_obs_1, weights_calib_inter_1, scores_1)
                offset_1_list.append(offset_1[:n_fold_one_1])

                y0_l = self.models_l_0[j][i].predict(X_calib_inter_0_fold_one)
                y0_u = self.models_u_0[j][i].predict(X_calib_inter_0_fold_one)
                y0_l_list.append(y0_l)
                y0_u_list.append(y0_u)

                y1_l = self.models_l_1[j][i].predict(X_calib_inter_1_fold_one)
                y1_u = self.models_u_1[j][i].predict(X_calib_inter_1_fold_one)
                y1_l_list.append(y1_l)
                y1_u_list.append(y1_u)

            y0_l = np.median(np.array(y0_l_list), axis=0) - np.median(np.array(offset_0_list), axis=0)
            y0_u = np.median(np.array(y0_u_list), axis=0) + np.median(np.array(offset_0_list), axis=0)
            y1_l = np.median(np.array(y1_l_list), axis=0) - np.median(np.array(offset_1_list), axis=0)
            y1_u = np.median(np.array(y1_u_list), axis=0) + np.median(np.array(offset_1_list), axis=0)

            C_calib_u_0.append(y0_u)
            C_calib_u_1.append(y1_u)
            C_calib_l_0.append(y0_l)
            C_calib_l_1.append(y1_l)

            X_calib_inter_0_fold_one_list.append(X_calib_inter_0_fold_one)
            X_calib_inter_1_fold_one_list.append(X_calib_inter_1_fold_one)
            X_calib_inter_0_fold_two_list.append(X_calib_inter_0_fold_two)
            X_calib_inter_1_fold_two_list.append(X_calib_inter_1_fold_two)
            Y_calib_inter_0_fold_two_list.append(Y_calib_inter_0_fold_two)
            Y_calib_inter_1_fold_two_list.append(Y_calib_inter_1_fold_two)
            
        X_calib_inter_0_fold_one_all = np.concatenate(X_calib_inter_0_fold_one_list, axis=0)
        X_calib_inter_1_fold_one_all = np.concatenate(X_calib_inter_1_fold_one_list, axis=0)
        X_calib_inter_0_fold_two_all = np.concatenate(X_calib_inter_0_fold_two_list, axis=0)
        X_calib_inter_1_fold_two_all = np.concatenate(X_calib_inter_1_fold_two_list, axis=0)
        Y_calib_inter_0_fold_two_all = np.concatenate(Y_calib_inter_0_fold_two_list, axis=0)
        Y_calib_inter_1_fold_two_all = np.concatenate(Y_calib_inter_1_fold_two_list, axis=0)

        C_calib_u_0 = np.concatenate(C_calib_u_0, axis=0)
        C_calib_u_1 = np.concatenate(C_calib_u_1, axis=0)
        C_calib_l_0 = np.concatenate(C_calib_l_0, axis=0)
        C_calib_l_1 = np.concatenate(C_calib_l_1, axis=0)

        # use fold one of calib_int data to fit regression on y_u and y_l
        self.C0_l_model.fit(X_calib_inter_0_fold_one_all, C_calib_l_0)
        self.C0_u_model.fit(X_calib_inter_0_fold_one_all, C_calib_u_0)
        self.C1_l_model.fit(X_calib_inter_1_fold_one_all, C_calib_l_1)
        self.C1_u_model.fit(X_calib_inter_1_fold_one_all, C_calib_u_1)

        scores_C0 = np.maximum(self.C0_l_model.predict(X_calib_inter_0_fold_two_all) - Y_calib_inter_0_fold_two_all, 
                               Y_calib_inter_0_fold_two_all - self.C0_u_model.predict(X_calib_inter_0_fold_two_all))
        offset_C0 = utils.standard_conformal(alpha, scores_C0)

        scores_C1 = np.maximum(self.C1_l_model.predict(X_calib_inter_1_fold_two_all) - Y_calib_inter_1_fold_two_all, 
                               Y_calib_inter_1_fold_two_all - self.C1_u_model.predict(X_calib_inter_1_fold_two_all))
        offset_C1 = utils.standard_conformal(alpha, scores_C1)

        # C0_test_l = self.C0_l_model.predict(X_test) - offset_C0
        # C0_test_u = self.C0_u_model.predict(X_test) + offset_C0
        # C1_test_l = self.C1_l_model.predict(X_test) - offset_C1
        # C1_test_u = self.C1_u_model.predict(X_test) + offset_C1

        return self.C0_l_model, self.C0_u_model, self.C1_l_model, self.C1_u_model
            
    def predict_counterfactual_naive(self, alpha, X_test, Y0, Y1):
        self.fit(method='naive')
        
        offset_0_list, offset_1_list = [] , []
        y0_l_list, y0_u_list = [], []
        y1_l_list, y1_u_list = [], []

        for j in range(self.n_folds):
            X_calib_inter_0 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==0, :]
            Y_calib_inter_0 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==0]
            X_calib_inter_1 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==1, :]
            Y_calib_inter_1 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==1]

            scores_0 = np.maximum(self.models_l_0[j].predict(X_calib_inter_0) - Y_calib_inter_0,
                                   Y_calib_inter_0 - self.models_u_0[j].predict(X_calib_inter_0))
            offset_0 = utils.standard_conformal(alpha, scores_0)
            offset_0_list.append(offset_0)

            scores_1 = np.maximum(self.models_l_1[j].predict(X_calib_inter_1) - Y_calib_inter_1,
                                   Y_calib_inter_1 - self.models_u_1[j].predict(X_calib_inter_1))
            offset_1 = utils.standard_conformal(alpha, scores_1)
            offset_1_list.append(offset_1)

            y1_l = self.models_l_1[j].predict(X_test)
            y1_u = self.models_u_1[j].predict(X_test)
            y0_l = self.models_l_0[j].predict(X_test)
            y0_u = self.models_u_0[j].predict(X_test)
            
            y0_l_list.append(y0_l)
            y0_u_list.append(y0_u)
            y1_l_list.append(y1_l)
            y1_u_list.append(y1_u)

        y0_l = np.median(np.array(y0_l_list), axis=0) - np.median(np.array(offset_0_list), axis=0)
        y0_u = np.median(np.array(y0_u_list), axis=0) + np.median(np.array(offset_0_list), axis=0)
        y1_l = np.median(np.array(y1_l_list), axis=0) - np.median(np.array(offset_1_list), axis=0)
        y1_u = np.median(np.array(y1_u_list), axis=0) + np.median(np.array(offset_1_list), axis=0)
        pause = True

        return y0_l, y0_u, y1_l, y1_u


    # def conformalize_naive_one_fold(self,
    #                        X_calib_int_0, Y_calib_int_0, 
    #                        X_calib_int_1,Y_calib_int_1,
    #                         j, alpha, 
    #                         ite_method="inexact"):
        
    #     # naive does not care about dr_use_Y
    #     self.C0_l_model_ = RandomForestRegressor()
    #     self.C0_u_model_ = RandomForestRegressor()
    #     self.C1_l_model_ = RandomForestRegressor()
    #     self.C1_u_model_ = RandomForestRegressor()

    #     # X_calib_obs_fold_one_0, X_calib_obs_fold_two_0, Y_calib_obs_fold_one_0, Y_calib_fold_obs_two_0 = train_test_split(
    #     #     X_calib_obs_0, Y_calib_obs_0, test_size=0.5, random_state=42)
    #     # X_calib_obs_fold_one_1, X_calib_obs_fold_two_1, Y_calib_obs_fold_one_1, Y_calib_fold_obs_two_1 = train_test_split(
    #     #     X_calib_obs_1, Y_calib_obs_1, test_size=0.5, random_state=42)

    #     # naive method only uses inter data

    #     X_calib_inter_fold_one_0, X_calib_inter_fold_two_0, Y_calib_inter_fold_one_0, Y_calib_inter_fold_two_0 = train_test_split(
    #         X_calib_int_0, Y_calib_int_0, test_size=0.5, random_state=42)
    #     X_calib_inter_fold_one_1, X_calib_inter_fold_two_1, Y_calib_inter_fold_one_1, Y_calib_inter_fold_two_1 = train_test_split(
    #         X_calib_int_1, Y_calib_int_1, test_size=0.5, random_state=42)
        
    #     scores_0 = np.maximum(self.models_l_0[j].predict(X_calib_inter_fold_one_0) - Y_calib_inter_fold_one_0,
    #                                Y_calib_inter_fold_one_0 - self.models_u_0[j].predict(X_calib_inter_fold_one_0))
    #     offset_0 = utils.standard_conformal(alpha, scores_0)
    #     # offset_0_list.append(offset_0)

    #     scores_1 = np.maximum(self.models_l_1[j].predict(X_calib_inter_fold_one_1) - Y_calib_inter_fold_one_1,
    #                                Y_calib_inter_fold_one_1 - self.models_u_1[j].predict(X_calib_inter_fold_one_1))
    #     offset_1 = utils.standard_conformal(alpha, scores_1)
    #     # offset_1_list.append(offset_1)

    #     # TODO: why should I use fold two data here?
    #     # predict counterfactuals
    #     y0_l = self.models_l_0[j].predict(X_calib_inter_fold_two_1) - offset_0
    #     y0_u = self.models_u_0[j].predict(X_calib_inter_fold_two_1) + offset_0

    #     y1_l = self.models_l_1[j].predict(X_calib_inter_fold_two_0) - offset_1
    #     y1_u = self.models_u_1[j].predict(X_calib_inter_fold_two_0) + offset_1

    #     # compute ITE
    #     # This is second line in Table 3 of Lei and Candes
    #     # Note that C1 is for the fold 2 of control group

    #     # y1_u = self.models_u_1[j][i].predict(X_calib_inter_fold_two_0) + offset_1
    #     # y1_l = self.models_l_1[j][i].predict(X_calib_inter_fold_two_0) - offset_1

    #     C1_u = y1_u - Y_calib_inter_fold_two_0
    #     C1_l = y1_l - Y_calib_inter_fold_two_0
        
    #     # This is first line in Table 3 of Lei and Candes
    #     # Note that C0 is for fold 2 of the treated group

    #     # y0_u = self.models_u_0[j][i].predict(X_calib_inter_fold_two_1) + offset_0
    #     # y0_l = self.models_l_0[j][i].predict(X_calib_inter_fold_two_1) - offset_0
        
    #     C0_u = Y_calib_inter_fold_two_1 - y0_l
    #     C0_l = Y_calib_inter_fold_two_1 - y0_u

    #     dummy_index = np.random.permutation(len(X_calib_inter_fold_two_0) + len(X_calib_inter_fold_two_1))

    #     if ite_method == "inexact":
    #         # ITE upper lower bound models
    #         self.tilde_C_ITE_model_l[j].fit(np.concatenate((X_calib_inter_fold_two_0, X_calib_inter_fold_two_1))[dummy_index, :],
    #                                     np.concatenate((C1_l, C0_l))[dummy_index])
                                        
    #         self.tilde_C_ITE_model_u[j].fit(np.concatenate((X_calib_inter_fold_two_0, X_calib_inter_fold_two_1))[dummy_index, :], 
    #                                         np.concatenate((C1_u, C0_u))[dummy_index])

    #     elif ite_method == "exact":
    #         C_l = np.concatenate((C1_l, C0_l))[dummy_index]
    #         C_u = np.concatenate((C1_u, C0_u))[dummy_index]
    #         X = np.concatenate((X_calib_inter_fold_two_0, 
    #                             X_calib_inter_fold_two_1))[dummy_index, :]
            
    #         X_train, X_calib, C_l_train, C_l_calib, C_u_train, C_u_calib = train_test_split(
    #             X, C_l, C_u, test_size=0.25, random_state=42)

    #         self.tilde_C_ITE_model_l[j].fit(X_train, C_l_train)                                                
    #         self.tilde_C_ITE_model_u[j].fit(X_train, C_u_train)

    #         # scores = np.maximum(C_u_calib - self.tilde_C_ITE_model_u[j].predict(X_calib), 
    #         #                         self.tilde_C_ITE_model_l[j].predict(X_calib) - C_l_calib)
    #         # offset = utils.standard_conformal(alpha, scores)
    #         # self.offset_list.append(offset)

    #         # calibrate upper lower bounds separately, treated as point estimate
    #         scores_u = np.abs(C_u_calib - self.tilde_C_ITE_model_u[j].predict(X_calib))
    #         scores_l = np.abs(C_l_calib - self.tilde_C_ITE_model_l[j].predict(X_calib)) 
            
    #         offset_u = utils.standard_conformal(alpha, scores_u)
    #         offset_l = utils.standard_conformal(alpha, scores_l)
            
    #         self.offset_u_list.append(offset_u)
    #         self.offset_l_list.append(offset_l)


    # def conformalize_one_fold(self, X_calib_obs_0, Y_calib_obs_0, X_calib_obs_1,
    #                        Y_calib_obs_1,
    #                        X_calib_int_0, Y_calib_int_0, X_calib_int_1,
    #                        Y_calib_int_1,
    #                         i, j, alpha, 
    #                         ite_method="inexact", 
    #                         cf_method="inexact",
    #                         dr_use_Y:int=1):

    #     """
    #     cf_method: naive or our methods: [inexact, exact]
    #     this is only used to replace the offsets computed by WCP in zonghao's code
        
    #     ite_method: naive or nested ite_methods from Lihua Lei's paper
    #     train models to predict upper lower bound of ITE using inexact method
    #     """

    #     # fold two of calib_obs is not used...
    #     X_calib_obs_fold_one_0, Y_calib_obs_fold_one_0 = X_calib_obs_0, Y_calib_obs_0
    #     X_calib_obs_fold_one_1, Y_calib_obs_fold_one_1 = X_calib_obs_1, Y_calib_obs_1

    #     # X_calib_obs_fold_one_0, X_calib_obs_fold_two_0, Y_calib_obs_fold_one_0, Y_calib_fold_obs_two_0 = train_test_split(
    #     #     X_calib_obs_0, Y_calib_obs_0, test_size=0.95, random_state=42)
    #     # X_calib_obs_fold_one_1, X_calib_obs_fold_two_1, Y_calib_obs_fold_one_1, Y_calib_fold_obs_two_1 = train_test_split(
    #     #     X_calib_obs_1, Y_calib_obs_1, test_size=0.95, random_state=42)

    #     X_calib_inter_fold_one_0, X_calib_inter_fold_two_0, Y_calib_inter_fold_one_0, Y_calib_inter_fold_two_0 = train_test_split(
    #         X_calib_int_0, Y_calib_int_0, test_size=0.5, random_state=42)
    #     X_calib_inter_fold_one_1, X_calib_inter_fold_two_1, Y_calib_inter_fold_one_1, Y_calib_inter_fold_two_1 = train_test_split(
    #         X_calib_int_1, Y_calib_int_1, test_size=0.5, random_state=42)

    #     # replace calibration data with calibration data fold one
    #     # replace test data with calibration data fold two
    #     # finally, use C_ITE on fold two to train quantile regression

    #     # if treatment = 1, we predict Y0
    #     # if treatment = 0, we predict Y1

    #     # different cf_method outputs different offsets
        
    #     if cf_method in ["inexact", "exact"]:
    #         # calibration, all use fold one

    #         D_calib_obs_fold_one_0, D_calib_inter_fold_one_0 = utils.get_dr_data(
    #                 X_calib_obs_fold_one_0, Y_calib_obs_fold_one_0, X_calib_inter_fold_one_0, Y_calib_inter_fold_one_0,
    #                   dr_use_Y, self.models_pseudo_label_0[j][i], train=False)
            
    #         D_calib_obs_fold_one_1, D_calib_inter_fold_one_1 = utils.get_dr_data(
    #                 X_calib_obs_fold_one_1, Y_calib_obs_fold_one_1, X_calib_inter_fold_one_1, Y_calib_inter_fold_one_1, 
    #                 dr_use_Y, self.models_pseudo_label_0[j][i], train=False)
           
    #         weights_calib_obs_fold_one_0 = self.density_models_0[j][i].compute_density_ratio(D_calib_obs_fold_one_0)
    #         weights_calib_inter_fold_one_0 = self.density_models_0[j][i].compute_density_ratio(D_calib_inter_fold_one_0)
    #         weights_calib_obs_fold_one_1 = self.density_models_1[j][i].compute_density_ratio(D_calib_obs_fold_one_1)
    #         weights_calib_inter_fold_one_1 = self.density_models_1[j][i].compute_density_ratio(D_calib_inter_fold_one_1)

    #         # calib with calib_obs data, reweighting use obs+int data
    #         # the same as inference in predict_counterfactual_inexact, except the data
    #         scores_0 = np.maximum(
    #             self.models_l_0[j][i].predict(X_calib_obs_fold_one_0) - Y_calib_obs_fold_one_0,
    #             Y_calib_obs_fold_one_0 - self.models_u_0[j][i].predict(X_calib_obs_fold_one_0))
    #         offset_0 = utils.weighted_conformal(alpha, 
    #                                             weights_calib_obs_fold_one_0, 
    #                                             weights_calib_inter_fold_one_0, 
    #                                             scores_0)[0]

    #         scores_1 = np.maximum(
    #             self.models_l_1[j][i].predict(X_calib_obs_fold_one_1) - Y_calib_obs_fold_one_1,
    #             Y_calib_obs_fold_one_1 - self.models_u_0[j][i].predict(X_calib_obs_fold_one_1))
    #         offset_1 = utils.weighted_conformal(alpha,
    #                                             weights_calib_obs_fold_one_1, 
    #                                             weights_calib_inter_fold_one_1, 
    #                                             scores_1)[0]

    #         # if cf_method == 'inexact':
    #         y0_l = self.models_l_0[j][i].predict(X_calib_inter_fold_two_1) - offset_0
    #         y0_u = self.models_u_0[j][i].predict(X_calib_inter_fold_two_1) + offset_0

    #         y1_l = self.models_l_1[j][i].predict(X_calib_inter_fold_two_0) - offset_1
    #         y1_u = self.models_u_1[j][i].predict(X_calib_inter_fold_two_0) + offset_1

    #         # if cf_method == 'exact':
    #         #     self.C0_l_model_ = RandomForestRegressor()
    #         #     self.C0_u_model_ = RandomForestRegressor()
    #         #     self.C1_l_model_ = RandomForestRegressor()
    #         #     self.C1_u_model_ = RandomForestRegressor()
    #         #     # will lead to a wider interval than inexact

    #         #     # TODO: further split the data into 3 folds
    #         #     # here we use factual to fit the QR models for PO intervals
    #         #     y0_l_f = self.models_l_0[j][i].predict(X_calib_inter_fold_two_0) - offset_0
    #         #     y0_u_f = self.models_u_0[j][i].predict(X_calib_inter_fold_two_0) + offset_0

    #         #     y1_l_f = self.models_l_1[j][i].predict(X_calib_inter_fold_two_1) - offset_1
    #         #     y1_u_f = self.models_u_1[j][i].predict(X_calib_inter_fold_two_1) + offset_1

    #         #     # use fold one of calib_int data to fit regression on y_u and y_l
    #         #     self.C0_l_model_.fit(X_calib_inter_fold_two_0, y0_l_f)
    #         #     self.C0_u_model_.fit(X_calib_inter_fold_two_0, y0_u_f)
    #         #     self.C1_l_model_.fit(X_calib_inter_fold_two_1, y1_l_f)
    #         #     self.C1_u_model_.fit(X_calib_inter_fold_two_1, y1_u_f)

    #         #     scores_C0 = np.maximum(self.C0_l_model.predict(X_calib_inter_fold_two_0) - Y_calib_inter_fold_two_0, 
    #         #                         Y_calib_inter_fold_two_0 - self.C0_u_model.predict(X_calib_inter_fold_two_0))
    #         #     offset_C0 = utils.standard_conformal(alpha, scores_C0)

    #         #     scores_C1 = np.maximum(self.C1_l_model.predict(X_calib_inter_fold_two_1) - Y_calib_inter_fold_two_1, 
    #         #                         Y_calib_inter_fold_two_1 - self.C1_u_model.predict(X_calib_inter_fold_two_1))
    #         #     offset_C1 = utils.standard_conformal(alpha, scores_C1)

    #         #     # override predicted upper lower bounds of cf outcomes
    #         #     # ideally, we should use fold 3 for this
    #         #     # X_calib_inter_fold_three_1 = X_calib_inter_fold_two_1
    #         #     # X_calib_inter_fold_three_0 = X_calib_inter_fold_two_0
    #         #     y0_l = self.C0_l_model_.predict(X_calib_inter_fold_two_1) - offset_C0
    #         #     y0_u = self.C0_u_model_.predict(X_calib_inter_fold_two_1) + offset_C0
    #         #     y1_l = self.C1_l_model_.predict(X_calib_inter_fold_two_0) - offset_C1
    #         #     y1_u = self.C1_u_model_.predict(X_calib_inter_fold_two_0) + offset_C1

    #     else:
    #         raise ValueError("cf_method has to be in [inexact, exact]")
    #     # compute ITE
    #     # This is second line in Table 3 of Lei and Candes
    #     # Note that C1 is for the fold 2 of control group

    #     # y1_u = self.models_u_1[j][i].predict(X_calib_inter_fold_two_0) + offset_1
    #     # y1_l = self.models_l_1[j][i].predict(X_calib_inter_fold_two_0) - offset_1

    #     # we use interventional data here because it is iid to test data
    #     # bounds for ITE, here C1 means the observed treatment is 1 for the unit
    #     C1_u = y1_u - Y_calib_inter_fold_two_0
    #     C1_l = y1_l - Y_calib_inter_fold_two_0
        
    #     # This is first line in Algo 3 of Lei and Candes
    #     # Note that C0 is for fold 2 of the treated group
        
    #     # C_ITE for T=1 in calib_inter_fold_two
    #     C0_u = Y_calib_inter_fold_two_1 - y0_l
    #     C0_l = Y_calib_inter_fold_two_1 - y0_u

    #     dummy_index = np.random.permutation(len(X_calib_inter_fold_two_0) + len(X_calib_inter_fold_two_1))

    #     if ite_method == "inexact":
    #         # C_l means lower bound of ITE
    #         self.tilde_C_ITE_model_l[j][i].fit(np.concatenate((X_calib_inter_fold_two_0, X_calib_inter_fold_two_1))[dummy_index, :],
    #                                     np.concatenate((C1_l, C0_l))[dummy_index])
                                        
    #         self.tilde_C_ITE_model_u[j][i].fit(np.concatenate((X_calib_inter_fold_two_0, X_calib_inter_fold_two_1))[dummy_index, :], 
    #                                         np.concatenate((C1_u, C0_u))[dummy_index])


    #     elif ite_method == "exact":
    #         C_l = np.concatenate((C1_l, C0_l))[dummy_index]
    #         C_u = np.concatenate((C1_u, C0_u))[dummy_index]
    #         X = np.concatenate((X_calib_inter_fold_two_0, 
    #                             X_calib_inter_fold_two_1))[dummy_index, :]
            
    #         X_train, X_calib, C_l_train, C_l_calib, C_u_train, C_u_calib = train_test_split(
    #             X, C_l, C_u, test_size=0.5, random_state=42)

    #         self.tilde_C_ITE_model_l[j][i].fit(X_train, C_l_train)                                                
    #         self.tilde_C_ITE_model_u[j][i].fit(X_train, C_u_train)

    #         # scores = np.maximum(C_u_calib - self.tilde_C_ITE_model_u[j][i].predict(X_calib), 
    #                                 # self.tilde_C_ITE_model_l[j][i].predict(X_calib) - C_l_calib)

    #         # calibrate upper lower bounds separately, treated as point estimate
    #         scores_u = np.abs(C_u_calib - self.tilde_C_ITE_model_u[j][i].predict(X_calib))
    #         scores_l = np.abs(C_l_calib - self.tilde_C_ITE_model_l[j][i].predict(X_calib))
            
    #         offset_u = utils.standard_conformal(alpha, scores_u)
    #         offset_l = utils.standard_conformal(alpha, scores_l)
            
    #         self.offset_u_list.append(offset_u)
    #         self.offset_l_list.append(offset_l)
            

    # def conformalize(self, alpha, 
    #                  ite_method,
    #                  C0_l, C0_u, C1_l, C1_u):
    #     """
    #     Calibrate the predictions of the meta-learner using standard conformal prediction
    #     """
        
    #     if ite_method == 'naive':   
    #         # Nothing needs to be done.
    #         return

    #     if ite_method == 'inexact':
    #         self.offset_u_list = []
    #         self.offset_l_list = []

    #         print(f'ite_method: {ite_method}, cf_method: {cf_method}')
    #         print('conformalizing for ITE intervals...')
            
    #         for j in tqdm(range(self.n_folds)):
    #             X_calib_inter_0 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==0, :]
    #             Y_calib_inter_0 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==0]
    #             X_calib_inter_1 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==1, :]
    #             Y_calib_inter_1 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==1]

    #             self.conformalize_naive_one_fold(
    #                     X_calib_inter_0, Y_calib_inter_0, 
    #                     X_calib_inter_1, Y_calib_inter_1,
    #                     j, alpha, 
    #                     ite_method=ite_method)
                    
    #             pause = True
                
    #             # if ite_method == "exact":
    #             #     self.offset_list = np.array(self.offset_list).reshape(self.n_folds,self.n_folds)

    #     elif cf_method in ['inexact', 'exact']:
                            
    #         self.offset_u_list = []
    #         self.offset_l_list = []

    #         print(f'ite_method: {ite_method}, cf_method: {cf_method}')
    #         print('conformalizing for ITE intervals...')
    #         for j in tqdm(range(self.n_folds)):
    #             X_calib_inter_0 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==0, :]
    #             Y_calib_inter_0 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==0]
    #             X_calib_inter_1 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==1, :]
    #             Y_calib_inter_1 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==1]

    #             for i in range(self.n_folds):
    #                 X_calib_obs_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
    #                 Y_calib_obs_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
    #                 X_calib_obs_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
    #                 Y_calib_obs_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

    #                 self.conformalize_one_fold(X_calib_obs_0, Y_calib_obs_0, 
    #                                             X_calib_obs_1, Y_calib_obs_1,
    #                                             X_calib_inter_0, Y_calib_inter_0, 
    #                                             X_calib_inter_1, Y_calib_inter_1, 
    #                                                 i, j, alpha, 
    #                                                 ite_method=ite_method,
    #                                                 cf_method=cf_method, 
    #                                                 dr_use_Y=dr_use_Y)
                
    #             pause = True
            
    #         if ite_method == "exact":
    #             self.offset_l_list = np.array(self.offset_l_list).reshape(self.n_folds,self.n_folds)
    #             self.offset_u_list = np.array(self.offset_u_list).reshape(self.n_folds,self.n_folds)

                
    #     else:
    #         raise ValueError('method must be one of naive, nested_inexact, nested_exact')

    # def predict_ITE(self, alpha, X_test, C0, C1, 
    #                 ite_method='naive', 
    #                 cf_method='naive'):
    #     """
    #     Interval-valued prediction of ITEs

    #     :param X: covariates of the test point

    #     outputs >> point estimate, lower bound and upper bound

    #     """
    #     if cf_method == 'naive':
    #         CI_ITE_l, CI_ITE_u = self.predict_ITE_naive(alpha, X_test, ite_method, C0, C1)
    #     elif cf_method in ['exact', 'inexact']:
    #         CI_ITE_l, CI_ITE_u = self.predict_ITE_ours(alpha, X_test, ite_method, C0, C1)

    #     return CI_ITE_l, CI_ITE_u
    
    # def predict_ITE_naive(self, alpha, X_test, ite_method, C0, C1):
    #     # predict ITE when cf_method = "naive"
    #     # which only have n_folds on inter data
    #     if ite_method == 'naive':
    #         # C0, C1 = self.predict_counterfactuals(alpha, X_test)
    #         return C1[0] - C0[1], C1[1] - C0[0]
        
    #     elif ite_method in ['inexact', 'exact']:
    #         CI_ITE_l_list, CI_ITE_u_list = [], []
    #         for i in range(self.n_folds):
    #             CI_ITE_l = self.tilde_C_ITE_model_l[i].predict(X_test)
    #             CI_ITE_u = self.tilde_C_ITE_model_u[i].predict(X_test)
                
    #             if ite_method == 'exact':
    #                 #TODO: this seems not correct, the offset should use weights for the test
    #                 CI_ITE_l -= self.offset_l_list[i]
    #                 CI_ITE_u += self.offset_u_list[i]

    #             CI_ITE_l_list.append(CI_ITE_l)
    #             CI_ITE_u_list.append(CI_ITE_u)
    #         CI_ITE_l = np.median(np.array(CI_ITE_l_list), axis=0)
    #         CI_ITE_u = np.median(np.array(CI_ITE_u_list), axis=0)
    #         return CI_ITE_l, CI_ITE_u
    #     else:
    #         raise ValueError('ite_method must be in [naive, inexact, exact]')


    # def predict_ITE_ours(self, alpha, X_test, ite_method, C0, C1):
    #     # predict ITE when cf_method in ["exact", "inexact"]
    #     # which only have n_folds on inter data
    #     if ite_method == 'naive':
    #         # C0, C1 = self.predict_counterfactuals(alpha, X_test)
    #         return C1[0] - C0[1], C1[1] - C0[0]
        
    #     elif ite_method in ['inexact', 'exact']:
    #         CI_ITE_l_list, CI_ITE_u_list = [], []
    #         for j in range(self.n_folds):
    #             for i in range(self.n_folds):
    #                 CI_ITE_l = self.tilde_C_ITE_model_l[j][i].predict(X_test)
    #                 CI_ITE_u = self.tilde_C_ITE_model_u[j][i].predict(X_test)
                    
    #                 if ite_method == 'exact':
    #                     CI_ITE_l -= self.offset_l_list[j][i]
    #                     CI_ITE_u += self.offset_u_list[j][i]

    #                 CI_ITE_l_list.append(CI_ITE_l)
    #                 CI_ITE_u_list.append(CI_ITE_u)
    #         CI_ITE_l = np.median(np.array(CI_ITE_l_list), axis=0)
    #         CI_ITE_u = np.median(np.array(CI_ITE_u_list), axis=0)
    #         return CI_ITE_l, CI_ITE_u
    #     else:
    #         raise ValueError('ite_method must be in [naive, inexact, exact]')


class TCP(BaseCP):
    def __init__(self, data_obs, data_inter, n_folds,
                 alpha=0.1, base_learner:str="GBM", quantile_regression:bool=False, K:int = 10,
                 density_ratio_model="MLP", seed=1, n_estimators:int=10):

        """
        Transductive conformal prediction, our method for Theorem 1

            :param n_folds: the number of folds for the DR learner cross-fitting (See [1])
            :param alpha: the target miscoverage level. alpha=.1 means that target coverage is 90%
            :param base_learner: the underlying regression model
                                - current options: ["GBM": gradient boosting machines, "RF": random forest]
            :param quantile_regression: Boolean for indicating whether the base learner is a quantile regression model
                                        or a point estimate of the CATE function. 

            :param K: number of bins for discretized Y
            :param DR_model: "DR" (traditional density ratio estimator) or "MLP" (MLP classifier classifying obs and int)

        """
        super().__init__(data_obs, data_inter, n_folds, alpha, base_learner, quantile_regression, n_estimators)
        self.K = K
        self.models = {}
        self.density_models = {}
        self.density_ratio_model = density_ratio_model
        self.seed = seed
        self.n_estimators = n_estimators

        X_inter = self.data_inter.filter(like = 'X').values
        T_inter = self.data_inter['T'].values
        Y_inter = self.data_inter['Y'].values

        X_obs = self.data_obs.filter(like = 'X').values
        T_obs = self.data_obs['T'].values
        Y_obs = self.data_obs['Y'].values

        self.X_inter_data = {}
        self.X_inter_data['0'] = X_inter[T_inter==0, :]
        self.X_inter_data['1'] = X_inter[T_inter==1, :]

        self.Y_inter_data = {}
        self.Y_inter_data['0'] = Y_inter[T_inter==0]
        self.Y_inter_data['1'] = Y_inter[T_inter==1]

        self.X_obs_data = {}
        self.X_obs_data['0'] = X_obs[T_obs==0, :]
        self.X_obs_data['1'] = X_obs[T_obs==1, :]

        self.Y_obs_data = {}
        self.Y_obs_data['0'] = Y_obs[T_obs==0]
        self.Y_obs_data['1'] = Y_obs[T_obs==1]

        # now, Y_bins = n_estimators
        self.Y_hat = np.linspace(np.min(Y_inter), np.max(Y_inter), n_estimators)

    # def data_preproc(self, X_test, T):

    #     i = j = 0

    #     # random select one fold
    #     while i == j:
    #         j = random.randint(0,self.n_folds-1)
    #         i = random.randint(0,self.n_folds-1)

    #     # here, I allow hat_y to be max value of Y by setting self.K+1 values for hat_y
    #     # Need n_fold ** 2 * n_test * K models...
    #     X_train_obs = self.X_train_obs_list[i][self.T_train_obs_list[i]==T, :]
    #     Y_train_obs = self.Y_train_obs_list[i][self.T_train_obs_list[i]==T]

    #     X_train_inter = self.X_train_inter_list[j][self.T_train_inter_list[j]==T, :]
    #     Y_train_inter = self.Y_train_inter_list[j][self.T_train_inter_list[j]==T]
    #     n_test_inter = X_test.shape[0]

    #     # standize X
    #     scaler = preprocessing.StandardScaler().fit(X_train_obs)
    #     X_train_obs = scaler.transform(X_train_obs)
    #     X_train_inter = scaler.transform(X_train_inter)
                
    #     # discretize Y
    #     Y_train_all = np.concatenate([Y_train_obs, Y_train_inter])

    #     n_train_obs = len(Y_train_obs)

    #     Y_train_all_dis, Y_bins = pd.qcut(Y_train_all, q=self.K, 
    #                                       labels=False, retbins=True, duplicates='drop')

    #     Y_train_obs_dis = Y_train_all_dis[:n_train_obs]
    #     Y_train_inter_dis = Y_train_all_dis[n_train_obs:]

    #     # train the density ratio estimation model with discretized labels
    #     D_obs = np.concatenate((X_train_obs, Y_train_obs_dis[:, None]), axis=1)
    #     D_inter = np.concatenate((X_train_inter, Y_train_inter_dis[:, None]), axis=1)

    #     return n_test_inter, D_obs, D_inter, X_train_obs, Y_train_obs, Y_bins


    def init_models(self, T):
        if self.quantile_regression:
            assert(self.base_learner)
            self.models[T] = {}
            self.models[T]["upper"] = base_learners_dict[self.base_learner](**self.first_CQR_args_u)
            self.models[T]["lower"] = base_learners_dict[self.base_learner](**self.first_CQR_args_l)
        else:
            # for each test sample, each random split
            self.models[T] = base_learners_dict[self.base_learner](**self.first_CQR_args)
            pass
            

    def train_density_model(self, D_inter, D_obs):
        if self.density_ratio_model == "DR": # density ratio estimator
            density_model = densratio(D_inter, D_obs, alpha=0.01)
            # self.density_models = density_model # save density ratio model
            weights_train = density_model.compute_density_ratio(D_obs)

        elif self.density_ratio_model == "MLP":

            density_model = MLPClassifier(random_state=self.seed, max_iter=100)

            # Assigning labels
            Y_obs_mlp = np.ones(len(D_obs))  # Label 1 for observed class
            Y_inter_mlp = np.zeros(len(D_inter))  # Label 0 for interventional class

            # Merging the datasets
            X_mlp = np.concatenate((D_obs, D_inter))
            Y_mlp = np.concatenate((Y_obs_mlp, Y_inter_mlp))

            density_model.fit(X_mlp, Y_mlp)
            
            # self.density_models[T] = density_model

            p_obs = density_model.predict_proba(D_obs)[:,1]

            weights_train = (1. - p_obs) / p_obs #TODO: double check

        return density_model, weights_train

    def predict_counterfactual(self, X_test, T, Y0, Y1):
        # Fit regression models for T=1 or T=0 on obs + inter data (x_1,y_1,...,x_M+1,y) for all y \in \mathcal{Y}

        n_test = X_test.shape[0]

        self.init_models(T)

        D_inter = np.concatenate((self.X_inter_data[f'{T}'], self.Y_inter_data[f'{T}'][:, None]), axis=1)
        D_obs = np.concatenate((self.X_obs_data[f'{T}'], self.Y_obs_data[f'{T}'][:, None]), axis=1)
        density_model, weights_train = self.train_density_model(D_inter, D_obs)

        # save results
        y_test_min = np.zeros(n_test)
        y_test_max = np.zeros(n_test)

        model_u = base_learners_dict[self.base_learner](**self.first_CQR_args_u)
        model_l = base_learners_dict[self.base_learner](**self.first_CQR_args_l)

        for test_idx in tqdm(range(X_test.shape[0])):
            x_test = X_test[test_idx, :][None, :]
            X_aug = np.concatenate((self.X_obs_data[f'{T}'], x_test), axis=0)
            y_interval = []

            def fit_model(y):
                Y_aug = np.concatenate((self.Y_obs_data[f'{T}'], np.array([y])), axis=0)
                model_u_ = clone(model_u)
                model_u_.fit(X_aug, Y_aug)
                model_l_ = clone(model_l)
                model_l_.fit(X_aug, Y_aug)

                Y_hat_l = model_l_.predict(X_aug)
                Y_hat_u = model_u_.predict(X_aug)
                scores = np.maximum(Y_hat_l - Y_aug, Y_aug - Y_hat_u)

                D_test = np.concatenate((x_test, np.array([y])[:, None]), axis=1)

                if self.density_ratio_model == "MLP":
                    p_obs = density_model.predict_proba(D_test)[:,1]
                    weight_test = (1. - p_obs) / p_obs #TODO: double check
                elif self.density_ratio_model == "DR":
                    weight_test = density_model.compute_density_ratio(D_test)

                offset = utils.weighted_transductive_conformal(
                    self.alpha, weights_train, weight_test, scores)
                return offset, scores[-1]

            # Parallelization
            results = Parallel(n_jobs=self.n_estimators)(delayed(fit_model)(y) for y in self.Y_hat)
            for i, y_hat in enumerate(self.Y_hat):
                if results[i][1] < results[i][0]:
                    y_interval.append(y_hat)

            y_test_min[test_idx] = min(y_interval)
            y_test_max[test_idx] = max(y_interval)

            # print(f"Interval is from {y_test_min[test_idx]} to {y_test_max[test_idx]}.") 

            pause = True
        return y_test_min, y_test_max
                


        

