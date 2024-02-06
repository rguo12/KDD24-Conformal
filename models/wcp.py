# partially based on code from Ahmed Alaa et al.

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from quantile_forest import RandomForestQuantileRegressor
from densratio import densratio

import numpy as np
import models.utils as utils

from iDCF.utils import plot_vec_dist

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from tqdm import tqdm

# Global options for baselearners (see class attributes below)

base_learners_dict = dict({"GBM": GradientBoostingRegressor, 
                           "RF": RandomForestQuantileRegressor})


class WCP:

    """

    """

    def __init__(self, data_obs, n_folds=5, alpha=0.1, base_learner="GBM", quantile_regression=True):

        """
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
            first_CQR_args_u = dict({"loss": "quantile", "alpha":1 - (self.alpha / 2), "n_estimators": n_estimators_target}) 
            first_CQR_args_l = dict({"loss": "quantile", "alpha":self.alpha/2, "n_estimators": n_estimators_target}) 
        elif self.base_learner == "RF":
            first_CQR_args_u = dict({"default_quantiles":1 - (self.alpha/2), "n_estimators": n_estimators_target})
            first_CQR_args_l = dict({"default_quantiles":self.alpha/2, "n_estimators": n_estimators_target})
        else:
            raise ValueError('base_learner must be one of GBM, RF')
        
        self.models_u_0 = [base_learners_dict[self.base_learner](**first_CQR_args_u) for _ in range(self.n_folds)]
        self.models_l_0 = [base_learners_dict[self.base_learner](**first_CQR_args_l) for _ in range(self.n_folds)] 
        self.models_u_1 = [base_learners_dict[self.base_learner](**first_CQR_args_u) for _ in range(self.n_folds)]
        self.models_l_1 = [base_learners_dict[self.base_learner](**first_CQR_args_l) for _ in range(self.n_folds)] 

        self.pscores_models = [LogisticRegression() for _ in range(self.n_folds)]

        if self.base_learner == "GBM":
            second_CQR_args_u = dict({"loss": "quantile", "alpha":0.6, "n_estimators": n_estimators_target})
            second_CQR_args_l = dict({"loss": "quantile", "alpha":0.4, "n_estimators": n_estimators_target})

        elif self.base_learner == "RF":
            second_CQR_args_u = dict({"default_quantiles":0.6, "n_estimators": n_estimators_target}) 
            second_CQR_args_l = dict({"default_quantiles":0.4, "n_estimators": n_estimators_target})
        
        self.tilde_C_ITE_model_u = [base_learners_dict[self.base_learner](**second_CQR_args_u) for _ in range(self.n_folds)] 
        self.tilde_C_ITE_model_l = [base_learners_dict[self.base_learner](**second_CQR_args_l) for _ in range(self.n_folds)] 

        self.data_obs = data_obs
        
        # initialize the lists for storing the cross-fitting indices
        self.train_obs_index_list = []
        self.calib_obs_index_list = []
        for _ in range(self.n_folds):
            self.train_obs_index_list.append(np.random.permutation(data_obs.index)[:int(0.75 * len(data_obs.index))])
            self.calib_obs_index_list.append(np.random.permutation(data_obs.index)[int(0.75 * len(data_obs.index)):])   
        
        self.X_train_obs_list, self.T_train_obs_list, self.Y_train_obs_list = [], [], []
        self.X_calib_obs_list, self.T_calib_obs_list, self.Y_calib_obs_list = [], [], []
        for i in range(self.n_folds):
            self.X_train_obs_list.append(data_obs.loc[self.train_obs_index_list[i]].filter(like = 'X').values)
            self.T_train_obs_list.append(data_obs.loc[self.train_obs_index_list[i]]['T'].values)
            self.Y_train_obs_list.append(data_obs.loc[self.train_obs_index_list[i]]['Y'].values)
            self.X_calib_obs_list.append(data_obs.loc[self.calib_obs_index_list[i]].filter(like = 'X').values)
            self.T_calib_obs_list.append(data_obs.loc[self.calib_obs_index_list[i]]['T'].values)
            self.Y_calib_obs_list.append(data_obs.loc[self.calib_obs_index_list[i]]['Y'].values)

        return

    def reset_tilde_C_ITE_models(self):

        n_estimators_target = 100
        if self.base_learner == "GBM":
            second_CQR_args_u = dict({"loss": "quantile", "alpha":0.6, "n_estimators": n_estimators_target})
            second_CQR_args_l = dict({"loss": "quantile", "alpha":0.4, "n_estimators": n_estimators_target})
        elif self.base_learner == "RF":
            second_CQR_args_u = dict({"default_quantiles":0.6, "n_estimators": n_estimators_target}) 
            second_CQR_args_l = dict({"default_quantiles":0.4, "n_estimators": n_estimators_target})
        
        self.tilde_C_ITE_model_u = [base_learners_dict[self.base_learner](**second_CQR_args_u) for _ in range(self.n_folds)] 
        self.tilde_C_ITE_model_l = [base_learners_dict[self.base_learner](**second_CQR_args_l) for _ in range(self.n_folds)] 

    def fit(self):
        """
        Fits the plug-in models and meta-learners using the sample (X, W, Y) and true propensity scores pscores
        """

        # loop over the cross-fitting folds
        for i in range(self.n_folds):
            X_train, T_train, Y_train = self.X_train_obs_list[i], self.T_train_obs_list[i], self.Y_train_obs_list[i]

            self.pscores_models[i].fit(X_train, T_train)
            self.models_u_0[i].fit(X_train[T_train==0, :], Y_train[T_train==0])
            self.models_l_0[i].fit(X_train[T_train==0, :], Y_train[T_train==0])
            self.models_u_1[i].fit(X_train[T_train==1, :], Y_train[T_train==1])
            self.models_l_1[i].fit(X_train[T_train==1, :], Y_train[T_train==1])

    def predict_counterfactuals(self, alpha, X_test):
        def weight_1(model, x):
            pscores = model.predict_proba(x)[:, 1]
            return 1. / pscores
        
        def weight_0(model, x):
            pscores = model.predict_proba(x)[:, 1]
            return 1. / (1.0 - pscores)
    
        y0_l_list, y0_u_list, y1_l_list, y1_u_list = [], [], [], []
        offset_0_list, offset_1_list = [], []

        for i in range(self.n_folds):
            X_calib_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
            X_calib_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
            Y_calib_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
            Y_calib_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

            Y1_calib_hat_u = self.models_u_1[i].predict(X_calib_1)
            Y1_calib_hat_l = self.models_l_1[i].predict(X_calib_1)

            weights_calib_1, weights_test_1, scores_1 = utils.weights_and_scores(weight_1, X_test, X_calib_1, Y_calib_1, 
                                                Y1_calib_hat_l, Y1_calib_hat_u, self.pscores_models[i])
            
            plot_vec_dist(scores_1, folder_name=f"dist_figs/cevae", 
                      filename=f'wcp_nonconf_score_cal_obs_1.png')
            
            plot_vec_dist(weights_calib_1, folder_name=f"dist_figs/cevae", 
                      filename=f'wcp_weights_cal_obs_1.png')

            plot_vec_dist(weights_test_1, folder_name=f"dist_figs/cevae", 
                      filename=f'wcp_weights_test_1.png')

            offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores_1)
            
            Y0_calib_hat_u = self.models_u_0[i].predict(X_calib_0)
            Y0_calib_hat_l = self.models_l_0[i].predict(X_calib_0)

            weights_calib_0, weights_test_0, scores_0 = utils.weights_and_scores(weight_0, X_test, X_calib_0, Y_calib_0, 
                                                Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models[i])

            plot_vec_dist(scores_0, folder_name=f"dist_figs/cevae", 
                      filename=f'wcp_nonconf_score_cal_obs_0.png')
            
            plot_vec_dist(weights_calib_0, folder_name=f"dist_figs/cevae", 
                      filename=f'wcp_weights_cal_obs_0.png')
            
            plot_vec_dist(weights_test_0, folder_name=f"dist_figs/cevae", 
                      filename=f'wcp_weights_test_0.png')

            offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores_0)

            offset_0_list.append(offset_0)
            offset_1_list.append(offset_1)

            y1_l = self.models_l_1[i].predict(X_test)
            y1_u = self.models_u_1[i].predict(X_test)
            y0_l = self.models_l_0[i].predict(X_test)
            y0_u = self.models_u_0[i].predict(X_test)
            
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
    
    # def predict_ITE(self, alpha, X_test, C0, C1, ite_method='naive'):
    #     """
    #     Interval-valued prediction of ITEs

    #     :param X: covariates of the test point

    #     outputs >> point estimate, lower bound and upper bound

    #     """
    #     if ite_method == 'naive':
    #         # C0, C1 = self.predict_counterfactuals(alpha, X_test)
    #         return C1[0] - C0[1], C1[1] - C0[0]
        
    #     elif ite_method == 'inexact':
    #         CI_ITE_l_list, CI_ITE_u_list = [], []
    #         for i in range(self.n_folds):
    #             CI_ITE_l = self.tilde_C_ITE_model_l[i].predict(X_test)
    #             CI_ITE_u = self.tilde_C_ITE_model_u[i].predict(X_test)
    #             CI_ITE_l_list.append(CI_ITE_l)
    #             CI_ITE_u_list.append(CI_ITE_u)
    #         CI_ITE_l = np.median(np.array(CI_ITE_l_list), axis=0)
    #         CI_ITE_u = np.median(np.array(CI_ITE_u_list), axis=0)
    #         return CI_ITE_l, CI_ITE_u
        
    #     elif ite_method == 'exact':
    #         CI_ITE_l_list, CI_ITE_u_list = [], []
    #         for i in range(self.n_folds):
    #             CI_ITE_l = self.tilde_C_ITE_model_l[i].predict(X_test) - self.offset_list[i]
    #             CI_ITE_u = self.tilde_C_ITE_model_u[i].predict(X_test) + self.offset_list[i]
    #             CI_ITE_l_list.append(CI_ITE_l)
    #             CI_ITE_u_list.append(CI_ITE_u)
    #         CI_ITE_l = np.median(np.array(CI_ITE_l_list), axis=0)
    #         CI_ITE_u = np.median(np.array(CI_ITE_u_list), axis=0)
    #         return CI_ITE_l, CI_ITE_u
    #     else:
    #         raise ValueError('method must be one of naive, nested_inexact, nested_exact')


    # def conformalize(self, alpha, ite_method='naive'):
    #     """
    #     Calibrate the predictions of the meta-learner using standard conformal prediction

    #     """
        
    #     if ite_method == 'naive':   
    #         # Nothing needs to be done.
    #         pass
        
    #     elif ite_method == 'inexact': 
    #         for i in range(self.n_folds):
    #             # calib obs data
    #             X_calib_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
    #             X_calib_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
    #             Y_calib_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
    #             Y_calib_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

    #             # calib_data = self.data_obs.loc[self.calib_index_list[i]]
    #             # for those T=0, we take its Y1.
    #             # Y1_calib_0 = calib_data[calib_data['T'] == 0]['Y1'].values
    #             # for those T=1, we take its Y0.
    #             # Y0_calib_1 = calib_data[calib_data['T'] == 1]['Y0'].values

    #             # X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, index_0, test_size=0.5, random_state=42)
    #             # X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, index_1, test_size=0.5, random_state=42)
                
    #             # split calib obs data into two
    #             # X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0, Y1_calib_fold_one_0, Y1_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, Y1_calib_0, test_size=0.5, random_state=42)
    #             # X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1, Y0_calib_fold_one_1, Y0_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, Y0_calib_1, test_size=0.5, random_state=42)

    #             X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0  = train_test_split(X_calib_0, Y_calib_0, test_size=0.5, random_state=42)
    #             X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, test_size=0.5, random_state=42)

    #             # predict upper lower bounds for fold 1 using main quantile regression models
    #             Y1_calib_hat_u = self.models_u_1[i].predict(X_calib_fold_one_1)
    #             Y1_calib_hat_l = self.models_l_1[i].predict(X_calib_fold_one_1)
                
    #             # get weighted conformal offset for the cf outcome on fold 2 (X_test is X_calib_fold_two_1 and X_calib_fold_two_0)
    #             def weight_fn_1(pscores_models, x):
    #                 pscores = pscores_models.predict_proba(x)[:, 1]
    #                 return (1.0 - pscores) / pscores

    #             # reweight scores of the treated to get offset of controlled
    #             weights_calib_1, weights_test_1, scores_1 = utils.weights_and_scores(weight_fn_1,
    #                                                                                  X_calib_fold_two_0, 
    #                                                                                  X_calib_fold_one_1, 
    #                                                                                  Y_calib_fold_one_1, 
    #                                                                                  Y1_calib_hat_l, 
    #                                                                                  Y1_calib_hat_u, 
    #                                                                                  self.pscores_models[i])

    #             # offset for the counterfactual outcome for the controlled group (fold 2)
    #             offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores_1)
            
    #             Y0_calib_hat_u = self.models_u_0[i].predict(X_calib_fold_one_0)
    #             Y0_calib_hat_l = self.models_l_0[i].predict(X_calib_fold_one_0)
            
    #             def weight_fn_0(pscores_models, x):
    #                 pscores = pscores_models.predict_proba(x)[:, 1]
    #                 return pscores / (1.0 - pscores)
            
    #             weights_calib_0, weights_test_0, scores_0 = utils.weights_and_scores(weight_fn_0, X_calib_fold_two_1, X_calib_fold_one_0, Y_calib_fold_one_0, 
    #                                                                            Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models[i])
                
    #             # offset for the counterfactual outcome for the treated group (fold 2)
    #             offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores_0)

    #             # ==================== Debug code ====================
    #             # u1 = utils.cross_fold_computation(self.models_u_1, X_calib_fold_two_0, proba=False) + offset_1
    #             # l1 = utils.cross_fold_computation(self.models_l_1, X_calib_fold_two_0, proba=False) - offset_1
    #             # coverage_1 = np.mean((Y1_calib_fold_two_0 >= l1) & (Y1_calib_fold_two_0 <= u1))
    #             # print('Debug: Coverage of Y(1) on second fold of calibration Y|T=1', coverage_1)
    #             # u0 = utils.cross_fold_computation(self.models_u_0, X_calib_fold_two_1, proba=False) + offset_0
    #             # l0 = utils.cross_fold_computation(self.models_l_0, X_calib_fold_two_1, proba=False) - offset_0
    #             # coverage_0 = np.mean((Y0_calib_fold_two_1 >= l0) & (Y0_calib_fold_two_1 <= u0))
    #             # print('Debug: Coverage of Y(0) on second fold of calibration Y|T=0', coverage_0)
    #             # pause = True

    #             # compute ITE
    #             # This is second line in Table 3 of Lei and Candes
    #             # Note that C1 is for the fold 2 of control group
    #             C1_u = (self.models_u_1[i].predict(X_calib_fold_two_0) + offset_1) - Y_calib_fold_two_0
    #             C1_l = (self.models_l_1[i].predict(X_calib_fold_two_0) - offset_1) - Y_calib_fold_two_0
                
    #             # This is first line in Table 3 of Lei and Candes
    #             # Note that C0 is for fold 2 of the treated group
    #             C0_u = Y_calib_fold_two_1 - (self.models_l_0[i].predict(X_calib_fold_two_1) - offset_0)
    #             C0_l = Y_calib_fold_two_1 - (self.models_u_0[i].predict(X_calib_fold_two_1) + offset_0)

    #             dummy_index = np.random.permutation(len(X_calib_fold_two_0) + len(X_calib_fold_two_1))
    #             self.tilde_C_ITE_model_l[i].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1))[dummy_index, :],
    #                                         np.concatenate((C1_l, C0_l))[dummy_index])
                                            
    #             self.tilde_C_ITE_model_u[i].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1))[dummy_index, :], 
    #                                         np.concatenate((C1_u, C0_u))[dummy_index])
    #             pause = True

    #     elif ite_method == 'exact':
    #         self.offset_list = []
    #         for i in range(self.n_folds):
    #             X_calib_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
    #             X_calib_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
    #             Y_calib_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
    #             Y_calib_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

    #             # calib_data = self.data_obs.loc[self.calib_index_list[i]]
    #             # Y1_calib_0 = calib_data[calib_data['T'] == 0]['Y1'].values
    #             # Y0_calib_1 = calib_data[calib_data['T'] == 1]['Y0'].values

    #             # X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, index_0, test_size=0.5, random_state=42)
    #             # X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, index_1, test_size=0.5, random_state=42)
                
    #             # X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0, Y1_calib_fold_one_0, Y1_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, Y1_calib_0, test_size=0.5, random_state=42)
    #             # X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1, Y0_calib_fold_one_1, Y0_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, Y0_calib_1, test_size=0.5, random_state=42)
    #             X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, test_size=0.5, random_state=42)
    #             X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, test_size=0.5, random_state=42)

    #             Y1_calib_hat_u = self.models_u_1[i].predict(X_calib_fold_one_1)
    #             Y1_calib_hat_l = self.models_l_1[i].predict(X_calib_fold_one_1)
                
    #             def weight_fn_1(pscores_models, x):
    #                 pscores = pscores_models.predict_proba(x)[:, 1]
    #                 return (1.0 - pscores) / pscores
            
    #             weights_calib_1, weights_test_1, scores_1 = utils.weights_and_scores(weight_fn_1, X_calib_fold_two_0, X_calib_fold_one_1, 
    #                                                                                  Y_calib_fold_one_1, 
    #                                                                                  Y1_calib_hat_l, Y1_calib_hat_u, self.pscores_models[i])
            
    #             offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores_1)
            
    #             Y0_calib_hat_u = self.models_u_0[i].predict(X_calib_fold_one_0)
    #             Y0_calib_hat_l = self.models_l_0[i].predict(X_calib_fold_one_0)
            
    #             def weight_fn_0(pscores_models, x):
    #                 pscores = pscores_models.predict_proba(x)[:, 1]
    #                 return pscores / (1.0 - pscores)
            
    #             weights_calib_0, weights_test_0, scores_0 = utils.weights_and_scores(weight_fn_0, X_calib_fold_two_1, X_calib_fold_one_0, Y_calib_fold_one_0, 
    #                                                                            Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models[i])
    #             offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores_0)

    #             # ==================== Debug code ====================
    #             # u1 = utils.cross_fold_computation(self.models_u_1, X_calib_fold_two_0, proba=False) + offset_1
    #             # l1 = utils.cross_fold_computation(self.models_l_1, X_calib_fold_two_0, proba=False) - offset_1
    #             # coverage_1 = np.mean((Y1_calib_fold_two_0 >= l1) & (Y1_calib_fold_two_0 <= u1))
    #             # print('Debug: Coverage of Y(1) on second fold of calibration Y|T=1', coverage_1)
    #             # u0 = utils.cross_fold_computation(self.models_u_0, X_calib_fold_two_1, proba=False) + offset_0
    #             # l0 = utils.cross_fold_computation(self.models_l_0, X_calib_fold_two_1, proba=False) - offset_0
    #             # coverage_0 = np.mean((Y0_calib_fold_two_1 >= l0) & (Y0_calib_fold_two_1 <= u0))
    #             # print('Debug: Coverage of Y(0) on second fold of calibration Y|T=0', coverage_0)
    #             # pause = True

    #             # This is second line in Table 3 of Lei and Candes
    #             # Note that C1 is for the control group
    #             C1_u = (self.models_u_1[i].predict(X_calib_fold_two_0) + offset_1) - Y_calib_fold_two_0
    #             C1_l = (self.models_l_1[i].predict(X_calib_fold_two_0) - offset_1) - Y_calib_fold_two_0
                
    #             # This is first line in Table 3 of Lei and Candes
    #             # Note that C0 is for the control group
    #             C0_u = Y_calib_fold_two_1 - (self.models_l_0[i].predict(X_calib_fold_two_1) - offset_0)
    #             C0_l = Y_calib_fold_two_1 - (self.models_u_0[i].predict(X_calib_fold_two_1) + offset_0)

    #             dummy_index = np.random.permutation(len(X_calib_fold_two_0) + len(X_calib_fold_two_1))

    #             C_l = np.concatenate((C1_l, C0_l))[dummy_index]
    #             C_u = np.concatenate((C1_u, C0_u))[dummy_index]
    #             X = np.concatenate((X_calib_fold_two_0,
    #                                  X_calib_fold_two_1))[dummy_index, :]
                
    #             X_train, X_calib, C_l_train, C_l_calib, C_u_train, C_u_calib = train_test_split(
    #                 X, C_l, C_u, test_size=0.25, random_state=42)

    #             self.tilde_C_ITE_model_l[i].fit(X_train, C_l_train)                                                
    #             self.tilde_C_ITE_model_u[i].fit(X_train, C_u_train)

    #             scores = np.maximum(C_u_calib - self.tilde_C_ITE_model_u[i].predict(X_calib), 
    #                                 self.tilde_C_ITE_model_l[i].predict(X_calib) - C_l_calib)
    #             offset = utils.standard_conformal(alpha, scores)
    #             self.offset_list.append(offset)
            
    #     else:
    #         raise ValueError('method must be one of naive, nested_inexact, nested_exact')
