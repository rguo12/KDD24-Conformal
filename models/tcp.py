from __future__ import absolute_import, division, print_function

import sys, os, time, random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from quantile_forest import RandomForestQuantileRegressor
# import density_ratio_estimation.src.densityratio as densityratio
from densratio import densratio
from concurrent.futures import ProcessPoolExecutor

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
                 alpha=0.1, base_learner="GBM", quantile_regression=True, n_estimators_target : int = 10):

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
                self.first_CQR_args_u = dict({"loss": "quantile", "alpha":1 - (self.alpha / 2), "n_estimators": n_estimators_target}) 
                self.first_CQR_args_l = dict({"loss": "quantile", "alpha":self.alpha/2, "n_estimators": n_estimators_target}) 
            elif self.base_learner == "RF":
                self.first_CQR_args_u = dict({"default_quantiles":1 - (self.alpha/2), "n_estimators": n_estimators_target})
                self.first_CQR_args_l = dict({"default_quantiles":self.alpha/2, "n_estimators": n_estimators_target})
            else:
                raise ValueError('base_learner must be one of GBM or RF')
            
        else:
            if self.base_learner == "GBM":
                self.first_CQR_args = dict({"loss": "squared_error", "n_estimators": n_estimators_target}) 
            elif self.base_learner == "RF":
                self.first_CQR_args = dict({"criterion": "squared_error", "n_estimators": n_estimators_target}) 
            else:
                raise ValueError('base_learner must be one of GBM or RF')

        self.data_obs = data_obs
        self.data_inter = data_inter
        self.train_obs_index_list, self.X_train_obs_list, self.T_train_obs_list, self.Y_train_obs_list, self.calib_obs_index_list, self.X_calib_obs_list, self.T_calib_obs_list, self.Y_calib_obs_list = utils.split_data(data_obs, n_folds, frac=0.75)
        self.train_inter_index_list, self.X_train_inter_list, self.T_train_inter_list, self.Y_train_inter_list, self.calib_inter_index_list, self.X_calib_inter_list, self.T_calib_inter_list, self.Y_calib_inter_list = utils.split_data(data_inter, n_folds, frac=0.75)      
        
        return

    def fit(self, method):
        # Implement the common fit logic here
        pass

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


    def fit(self, method):
        """
        Fits the plug-in models and meta-learners using the sample (X, W, Y) and true propensity scores pscores
        """
        if method == 'two_stage_inexact' or method == 'two_stage_exact':
            self.models_u_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_l_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_u_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_l_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)] 

            self.density_models_0 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.density_models_1 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]

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
                    X_train_obs_0 = self.X_train_obs_list[i][self.T_train_obs_list[i]==0, :]
                    Y_train_obs_0 = self.Y_train_obs_list[i][self.T_train_obs_list[i]==0]
                    X_train_obs_1 = self.X_train_obs_list[i][self.T_train_obs_list[i]==1, :]
                    Y_train_obs_1 = self.Y_train_obs_list[i][self.T_train_obs_list[i]==1]

                    self.models_u_0[j][i].fit(X_train_obs_0, Y_train_obs_0)
                    self.models_l_0[j][i].fit(X_train_obs_0, Y_train_obs_0)
                    self.models_u_1[j][i].fit(X_train_obs_1, Y_train_obs_1)
                    self.models_l_1[j][i].fit(X_train_obs_1, Y_train_obs_1)
                    
                    D_train_obs_0 = np.concatenate((X_train_obs_0, Y_train_obs_0[:, None]), axis=1)
                    D_train_inter_0 = np.concatenate((X_train_inter_0, Y_train_inter_0[:, None]), axis=1)
                    D_train_obs_1 = np.concatenate((X_train_obs_1, Y_train_obs_1[:, None]), axis=1)
                    D_train_inter_1 = np.concatenate((X_train_inter_1, Y_train_inter_1[:, None]), axis=1)
                    
                    self.density_models_0[j][i] = densratio(D_train_inter_0, D_train_obs_0, verbose=False, alpha=0.01)
                    self.density_models_1[j][i] = densratio(D_train_inter_1, D_train_obs_1, verbose=False, alpha=0.01)

        elif method == 'naive':
            self.models_u_0 = [base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)]
            self.models_l_0 = [base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)]
            self.models_u_1 = [base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)]
            self.models_l_1 = [base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)]

            for j in range(self.n_folds):
                X_train_inter_0 = self.X_train_inter_list[j][self.T_train_inter_list[j]==0, :]
                Y_train_inter_0 = self.Y_train_inter_list[j][self.T_train_inter_list[j]==0]
                X_train_inter_1 = self.X_train_inter_list[j][self.T_train_inter_list[j]==1, :]
                Y_train_inter_1 = self.Y_train_inter_list[j][self.T_train_inter_list[j]==1]

                self.models_u_0[j].fit(X_train_inter_0, Y_train_inter_0)
                self.models_l_0[j].fit(X_train_inter_0, Y_train_inter_0)
                self.models_u_1[j].fit(X_train_inter_1, Y_train_inter_1)
                self.models_l_1[j].fit(X_train_inter_1, Y_train_inter_1)

    def predict_counterfactual_inexact(self, alpha, X_test, Y0, Y1):
        print("Fitting models ... ")
        self.fit(method='two_stage_inexact')
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

                D_calib_obs_0 = np.concatenate((X_calib_obs_0, Y_calib_obs_0[:, None]), axis=1)
                D_calib_inter_0 = np.concatenate((X_calib_inter_0, Y_calib_inter_0[:, None]), axis=1)
                D_calib_obs_1 = np.concatenate((X_calib_obs_1, Y_calib_obs_1[:, None]), axis=1)
                D_calib_inter_1 = np.concatenate((X_calib_inter_1, Y_calib_inter_1[:, None]), axis=1)

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

        C0_test_l = self.C0_l_model.predict(X_test)
        C0_test_u = self.C0_u_model.predict(X_test)
        C1_test_l = self.C1_l_model.predict(X_test)
        C1_test_u = self.C1_u_model.predict(X_test)
        return C0_test_l, C0_test_u, C1_test_l, C1_test_u 
    
    def predict_counterfactual_exact(self, alpha, X_test, Y0, Y1):
        print("Fitting models ... ")
        self.fit(method='two_stage_exact')
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

            calib_num_0, calib_num_1 = len(X_calib_inter_0), len(X_calib_inter_1)
            X_calib_inter_0_fold_one, X_calib_inter_0_fold_two = X_calib_inter_0[:int(calib_num_0/2), :], X_calib_inter_0[int(calib_num_0/2):, :]
            Y_calib_inter_0_fold_one, Y_calib_inter_0_fold_two = Y_calib_inter_0[:int(calib_num_0/2)], Y_calib_inter_0[int(calib_num_0/2):]
            X_calib_inter_1_fold_one, X_calib_inter_1_fold_two = X_calib_inter_1[:int(calib_num_1/2), :], X_calib_inter_1[int(calib_num_1/2):, :]
            Y_calib_inter_1_fold_one, Y_calib_inter_1_fold_two = Y_calib_inter_1[:int(calib_num_1/2)], Y_calib_inter_1[int(calib_num_1/2):]

            offset_0_list, offset_1_list = [] , []
            y0_l_list, y0_u_list = [], []
            y1_l_list, y1_u_list = [], []

            for i in range(self.n_folds):
                X_calib_obs_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
                Y_calib_obs_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
                X_calib_obs_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
                Y_calib_obs_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

                D_calib_obs_0 = np.concatenate((X_calib_obs_0, Y_calib_obs_0[:, None]), axis=1)
                D_calib_inter_0 = np.concatenate((X_calib_inter_0_fold_one, Y_calib_inter_0_fold_one[:, None]), axis=1)
                D_calib_obs_1 = np.concatenate((X_calib_obs_1, Y_calib_obs_1[:, None]), axis=1)
                D_calib_inter_1 = np.concatenate((X_calib_inter_1_fold_one, Y_calib_inter_1_fold_one[:, None]), axis=1)

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

        C0_test_l = self.C0_l_model.predict(X_test) - offset_C0
        C0_test_u = self.C0_u_model.predict(X_test) + offset_C0
        C1_test_l = self.C1_l_model.predict(X_test) - offset_C1
        C1_test_u = self.C1_u_model.predict(X_test) + offset_C1
        return C0_test_l, C0_test_u, C1_test_l, C1_test_u 
    
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

            scores_0 = np.maximum(self.models_l_0[j].predict(X_calib_inter_0) - Y_calib_inter_0, Y_calib_inter_0 - self.models_u_0[j].predict(X_calib_inter_0))
            offset_0 = utils.standard_conformal(alpha, scores_0)
            offset_0_list.append(offset_0)

            scores_1 = np.maximum(self.models_l_1[j].predict(X_calib_inter_1) - Y_calib_inter_1, Y_calib_inter_1 - self.models_u_1[j].predict(X_calib_inter_1))
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


class TCP(BaseCP):
    def __init__(self, data_obs, data_inter, n_folds,
                 alpha=0.1, base_learner:str="GBM", quantile_regression:bool=False, K:int = 10,
                 density_ratio_model="MLP", seed=1):

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
        super().__init__(data_obs, data_inter, n_folds, alpha, base_learner, quantile_regression)
        self.K = K
        self.models = {}
        self.density_models = {}
        self.density_ratio_model = density_ratio_model
        self.seed = seed

    def data_preproc(self, X_test, T):
        # random select one fold
        # j = random.randint(0,self.n_folds-1)
        # i = random.randint(0,self.n_folds-1)

        # select the 1st fold
        i = j = 0

        X_train_inter = self.X_train_inter_list[j][self.T_train_inter_list[j]==T, :]
        Y_train_inter = self.Y_train_inter_list[j][self.T_train_inter_list[j]==T]
        n_test_inter = X_test.shape[0]
        
        # here, I allow hat_y to be max value of Y by setting self.K+1 values for hat_y
        # Need n_fold ** 2 * n_test * K models...
        X_train_obs = self.X_train_obs_list[i][self.T_train_obs_list[i]==T, :]
        Y_train_obs = self.Y_train_obs_list[i][self.T_train_obs_list[i]==T]
                
        # discretize Y
        Y_train_all = np.concatenate([Y_train_obs, Y_train_inter])

        n_train_obs = len(Y_train_obs)

        Y_train_all_dis, Y_bins = pd.qcut(Y_train_all, q=self.K, 
                                          labels=False, retbins=True, duplicates='drop')

        Y_train_obs_dis = Y_train_all_dis[:n_train_obs]
        Y_train_inter_dis = Y_train_all_dis[n_train_obs:]

        # train the density ratio estimation model with discretized labels
        D_obs = np.concatenate((X_train_obs, Y_train_obs_dis[:, None]), axis=1)
        D_inter = np.concatenate((X_train_inter, Y_train_inter_dis[:, None]), axis=1)

        return n_test_inter, D_obs, D_inter, X_train_obs, Y_train_obs, Y_bins


    def init_models(self, T, n_test_inter):
        if self.quantile_regression:
            assert(self.base_learner)
            self.models[T] = {}
            self.models[T]["upper"] = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.K+1)]
                                        for _ in range(n_test_inter)]
            self.models[T]["lower"] = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.K+1)]
                                        for _ in range(n_test_inter)]
            # raise NotImplementedError("Only support quantile_regression = False")

        else:
            # for each test sample, each random split
            self.models[T] = [[base_learners_dict[self.base_learner](**self.first_CQR_args) for _ in range(self.K+1)]
                                        for _ in range(n_test_inter)]
            pass
    
    def train_density_model(self, T, D_inter, D_obs):
        if self.density_ratio_model == "DR": # density ratio estimator

            density_model = densratio(D_inter, D_obs, alpha=0.01)
            self.density_models[T] = density_model # save density ratio model
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
            
            self.density_models[T] = density_model

            p_obs = density_model.predict_proba(D_obs)[:,1]

            weights_train = (1-p_obs)/p_obs #TODO: double check

        return density_model, weights_train

    def predict_counterfactual(self, X_test, T=1):
        # Fit regression models for T=1 or T=0 on obs + inter data (x_1,y_1,...,x_M+1,y) for all y \in \mathcal{Y}

        # self.density_models_0 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]

        # need n_fold ** 2 density models
        self.density_models = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]

        y_test_min = [] # n_test
        y_test_max = [] # n_test

        n_test_inter, D_obs, D_inter, X_train_obs, Y_train_obs, Y_bins = self.data_preproc(X_test,T)

        self.init_models(T, n_test_inter)

        density_model, weights_train = self.train_density_model(T, D_inter, D_obs)

        # save results
        y_test_min = np.zeros(n_test_inter)
        y_test_max = np.zeros(n_test_inter)

        update_interval = 50 # for tqdm

        with tqdm(total=X_test.shape[0]) as pbar:

            for test_idx in range(X_test.shape[0]):
                x_test = X_test[test_idx, :]
                X_train_test_mixed = np.concatenate((X_train_obs, x_test.reshape(1,-1)), axis=0)
                y_interval = []

                for k, y_hat in enumerate(Y_bins):
                    D_test = np.array(np.concatenate((x_test, np.array([y_hat])), axis=0)[None, :])

                    if self.density_ratio_model == "DR":
                        weight_test = self.density_models[T].compute_density_ratio(D_test)
                        
                    elif self.density_ratio_model == "MLP":
                        p_obs = density_model.predict_proba(D_test)[:,1]
                        weight_test = (1-p_obs)/p_obs #TODO: double check
                    
                    Y_train_test_mixed = np.concatenate((Y_train_obs, np.array([y_hat])), axis=0)
                    # self.fit(X_train_test_mixed, Y_train_test_mixed, T=T)

                    if self.quantile_regression:
                        model_u = self.models[T]["upper"][test_idx][k]
                        model_l = self.models[T]["lower"][test_idx][k]
                        model_u.fit(X_train_test_mixed, Y_train_test_mixed)
                        model_l.fit(X_train_test_mixed, Y_train_test_mixed)

                        Y_hat_l = model_l.predict(X_train_test_mixed)
                        Y_hat_u = model_u.predict(X_train_test_mixed)

                        scores = np.maximum(
                            Y_hat_l - Y_train_test_mixed, 
                            Y_train_test_mixed - Y_hat_u)
                        scores = np.maximum(scores, 0.0)

                    else:
                        model = self.models[T][test_idx][k]
                        model.fit(X_train_test_mixed, Y_train_test_mixed)

                        Y_hat = model.predict(X_train_test_mixed)
                        scores = np.abs(Y_hat - Y_train_test_mixed)

                    offset = utils.weighted_transductive_conformal(
                        self.alpha, weights_train, weight_test, scores)
                    
                    print("Scores[-1]:{}, Offset:{}".format(scores[-1], offset))
                    if scores[-1] < offset:
                        # compare test sample score with offset
                        # append to y_interval iff scores[-1] < offset
                        y_interval.append(float(y_hat))

                y_test_min[test_idx] = min(y_interval)
                y_test_max[test_idx] = max(y_interval)
                
                if test_idx % update_interval == 0:
                    pbar.update(update_interval)
                    
        return y_test_min, y_test_max
                


        

