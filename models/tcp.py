from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
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


class TCP:

    """

    """

    def __init__(self, data_obs, data_inter, n_folds, alpha=0.1, base_learner="GBM", quantile_regression=True):

        """
        Transductive conformal prediction

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
        n_estimators_nuisance = 100
        n_estimators_target = 100
        
        if self.base_learner == "GBM":
            self.first_CQR_args_u = dict({"loss": "quantile", "alpha":1 - (self.alpha / 2), "n_estimators": n_estimators_target}) 
            self.first_CQR_args_l = dict({"loss": "quantile", "alpha":self.alpha/2, "n_estimators": n_estimators_target}) 
        elif self.base_learner == "RF":
            self.first_CQR_args_u = dict({"default_quantiles":1 - (self.alpha/2), "n_estimators": n_estimators_target})
            self.first_CQR_args_l = dict({"default_quantiles":self.alpha/2, "n_estimators": n_estimators_target})
        else:
            raise ValueError('base_learner must be one of GBM or RF')

        self.data_obs = data_obs
        self.data_inter = data_inter
        self.train_obs_index_list, self.X_train_obs_list, self.T_train_obs_list, self.Y_train_obs_list, self.calib_obs_index_list, self.X_calib_obs_list, self.T_calib_obs_list, self.Y_calib_obs_list = utils.split_data(data_obs, n_folds, frac=0.75)
        self.train_inter_index_list, self.X_train_inter_list, self.T_train_inter_list, self.Y_train_inter_list, self.calib_inter_index_list, self.X_calib_inter_list, self.T_calib_inter_list, self.Y_calib_inter_list = utils.split_data(data_inter, n_folds, frac=0.75)      
        
        self.C0_l_model = RandomForestRegressor()
        self.C0_u_model = RandomForestRegressor()
        self.C1_l_model = RandomForestRegressor()
        self.C1_u_model = RandomForestRegressor()
        return

    def fit(self, method):
        """
        Fits the plug-in models and meta-learners using the sample (X, W, Y) and true propensity scores pscores
        """
        if method == 'nested':
            self.models_u_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_l_0 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_u_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_u) for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.models_l_1 = [[base_learners_dict[self.base_learner](**self.first_CQR_args_l) for _ in range(self.n_folds)] for _ in range(self.n_folds)] 

            self.density_models_0 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]
            self.density_models_1 = [[None for _ in range(self.n_folds)] for _ in range(self.n_folds)]

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

                    X_train_obs_inter_0 = np.concatenate((X_train_obs_0, X_train_inter_0), axis=0)
                    Y_train_obs_inter_0 = np.concatenate((Y_train_obs_0, Y_train_inter_0))
                    X_train_obs_inter_1 = np.concatenate((X_train_obs_1, X_train_inter_1), axis=0)
                    Y_train_obs_inter_1 = np.concatenate((Y_train_obs_1, Y_train_inter_1))

                    self.models_u_0[j][i].fit(X_train_obs_inter_0, Y_train_obs_inter_0)
                    self.models_l_0[j][i].fit(X_train_obs_inter_0, Y_train_obs_inter_0)
                    self.models_u_1[j][i].fit(X_train_obs_inter_1, Y_train_obs_inter_1)
                    self.models_l_1[j][i].fit(X_train_obs_inter_1, Y_train_obs_inter_1)
                    
                    D_train_obs_0 = np.concatenate((X_train_obs_0, Y_train_obs_0[:, None]), axis=1)
                    D_train_inter_0 = np.concatenate((X_train_inter_0, Y_train_inter_0[:, None]), axis=1)
                    D_train_obs_1 = np.concatenate((X_train_obs_1, Y_train_obs_1[:, None]), axis=1)
                    D_train_inter_1 = np.concatenate((X_train_inter_1, Y_train_inter_1[:, None]), axis=1)
                    
                    self.density_models_0[j][i] = densratio(D_train_inter_0, D_train_obs_0, verbose=False)
                    self.density_models_1[j][i] = densratio(D_train_inter_1, D_train_obs_1, verbose=False)
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
        self.fit(method='nested')
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
        self.fit(method='nested')
        print("Fitting models done. ")
        
        C_calib_u_0, C_calib_l_0 = [], []
        C_calib_u_1, C_calib_l_1 = [], []
        X_calib_inter_0_all, X_calib_inter_1_all = [], []

        for j in range(self.n_folds):
            X_calib_inter_0 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==0, :]
            Y_calib_inter_0 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==0]
            X_calib_inter_1 = self.X_calib_inter_list[j][self.T_calib_inter_list[j]==1, :]
            Y_calib_inter_1 = self.Y_calib_inter_list[j][self.T_calib_inter_list[j]==1]

            calib_num_0, calib_num_1 = len(X_calib_inter_0), len(X_calib_inter_1)
            X_calib_inter_0_fold_one, X_calib_inter_0_fold_two = X_calib_inter_0[:int(calib_num_0/2), :], X_calib_inter_0[int(calib_num_0/2):, :]
            Y_calib_inter_0_fold_one = Y_calib_inter_0[:int(calib_num_0/2)]

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