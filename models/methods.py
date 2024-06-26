import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from models.drlearner import *
from models.wcp import *
from models.tcp import *

from models.utils import plot_tsne, eval_po

def conformal_metalearner(df, metalearner="DR", quantile_regression=True, alpha=0.1, test_frac=0.1):
    
    if len(df)==2:
        
        train_data1, test_data = df
    
    else:
    
        train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", 
                                    quantile_regression=quantile_regression, 
                                    metalearner=metalearner) 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return coverage, average_interval_width, PEHE, conformity_scores


def weighted_conformal_prediction(df_o, quantile_regression, alpha, test_frac, target):
    """_summary_

    Args:
        df_o (_type_): _description_
        quantile_regression (_type_): _description_
        alpha (_type_): _description_
        test_frac (_type_): _description_
        target (_type_): _description_
        method (str): only used as name for the row in saved results

    Returns:
        _type_: _description_
    """
       
    if len(df_o)==2:
        
        train_data, test_data = df_o
    
    else:
    
        train_data, test_data = train_test_split(df_o, test_size=test_frac, random_state=42)

    X_test = test_data.filter(like = 'X').values
    T_test = test_data[['T']].values.reshape((-1,)) 
    Y_test = test_data[['Y']].values.reshape((-1,))
    ps_test = test_data[['ps']].values

    Y0, Y1 = test_data[['Y0']].values.reshape((-1,)), test_data[['Y1']].values.reshape((-1,))
    ITE_test = Y1 - Y0

    # if target == 'counterfactual':
    model = WCP(data_obs=train_data,
                alpha=alpha, 
                base_learner="RF", 
                quantile_regression=quantile_regression) 
    model.fit()
    # model.conformalize(alpha, method='naive')
    C0_l, C0_u, C1_l, C1_u = model.predict_counterfactuals(alpha, X_test)

    coverage_0 = np.mean((Y0 >= C0_l) & (Y0 <= C0_u))
    coverage_1 = np.mean((Y1 >= C1_l) & (Y1 <= C1_u))
    interval_width_0 = np.mean(np.abs(C0_u - C0_l))
    interval_width_1 = np.mean(np.abs(C1_u - C1_l))
    
    X_calib_inter_0 = df_o[df_o['T']==0].filter(like = 'X').values
    Y_calib_inter_0 = df_o[df_o['T']==0]['Y'].values
    X_calib_inter_1 = df_o[df_o['T']==1].filter(like = 'X').values
    Y_calib_inter_1 = df_o[df_o['T']==1]['Y'].values

    _, _, C1_l_calib, C1_u_calib = model.predict_counterfactuals(alpha, X_calib_inter_0)
    C0_l_calib, C0_u_calib, _, _ = model.predict_counterfactuals(alpha, X_calib_inter_1)

    res_list = []
    for ite_method in ["naive", "inexact", "exact"]:

        res = {}
        res['cf_method'] = 'WCP'
        res['ite_method'] = ite_method
        res['coverage_0'] = coverage_0
        res['coverage_1'] = coverage_1
        res['interval_width_0'] = interval_width_0
        res['interval_width_1'] = interval_width_1

        CI_ITE_l, CI_ITE_u = predict_ITE(alpha, X_test, C0_l, C0_u, C1_l, C1_u, X_calib_inter_0, X_calib_inter_1,   
                                        C0_l_calib, C0_u_calib, C1_l_calib, C1_u_calib, 
                                        Y_calib_inter_0, Y_calib_inter_1, ite_method)
        
        coverage_ITE = np.mean((ITE_test >= CI_ITE_l) & (ITE_test <= CI_ITE_u))
        interval_width_ITE = np.mean(np.abs(CI_ITE_u - CI_ITE_l))
        print('Coverage of ITE', coverage_ITE)
        print('Interval width of ITE', interval_width_ITE)

        res['coverage_ITE'] = coverage_ITE
        res['interval_width'] = interval_width_ITE
        res_list.append(res)
    return res_list


def run_conformal(df_o, df_i, 
                  quantile_regression, 
                  n_folds : int, alpha : float, test_frac : float, target : str, 
                  cf_method : str,
                #   ite_method : str,
                  density_ratio_model : str = "MLP",
                  base_learner : str = "RF",
                  n_estimators:int = 10,
                  K:int = 10,
                  plot:bool = False,
                  dataset:str = None,
                  dr_use_Y:int = 1,
                  seed:int=42):
    """_summary_
    Run naive CP on intervention, and our exact and inexact methods

    Args:
        df_o (_type_): observational data
        df_i (_type_): interventional data
        quantile_regression (_type_): _description_
        n_folds (_type_): _description_
        alpha (_type_): _description_
        test_frac (_type_): _description_
        target (_type_): select from counterfactual
        method (_type_): select from naive, inexact and exact

    Returns:
        _type_: _description_
    """

    if len(df_o)==2:
        
        train_data, test_data = df_o
    
    else:
        # test data is obs data, which does not matter as for test we consider ITE/CF Outcome
        train_data, test_data = train_test_split(df_o, test_size=test_frac, random_state=seed)

    X_test = test_data.filter(like = 'X').values
    T_test = test_data[['T']].values.reshape((-1,)) 
    Y_test = test_data[['Y']].values
    ps_test = test_data[['ps']].values

    D_test = np.concatenate((X_test, Y_test),axis=1)

    Y_test = Y_test.reshape(-1)

    Y0, Y1 = test_data[['Y0']].values.reshape((-1,)), test_data[['Y1']].values.reshape((-1,))
    ITE_test = Y1 - Y0

    # This is for ite evaluation
    X_calib_inter_0 = np.concatenate((df_i[df_i['T']==0].filter(like = 'X').values, df_o[df_o['T']==0].filter(like = 'X').values))
    Y_calib_inter_0 = np.concatenate((df_i[df_i['T']==0]['Y'].values, df_o[df_o['T']==0]['Y'].values), axis=0)
    X_calib_inter_1 = np.concatenate((df_i[df_i['T']==1].filter(like = 'X').values, df_o[df_o['T']==1].filter(like = 'X').values))
    Y_calib_inter_1 = np.concatenate((df_i[df_i['T']==1]['Y'].values, df_o[df_o['T']==1]['Y'].values), axis=0)

    print(f'cf_method = {cf_method}')
    if cf_method == 'naive':
        model = SplitCP(data_obs=train_data,
                    data_inter=df_i,
                    n_folds=n_folds,
                    alpha=alpha, 
                    base_learner=base_learner,
                    quantile_regression=quantile_regression)
        
        C0_l, C0_u, C1_l, C1_u = model.predict_counterfactual_naive(alpha, X_test, Y0, Y1)
        
        _, _, C1_l_calib, C1_u_calib = model.predict_counterfactual_naive(alpha, X_calib_inter_0, Y0, Y1)
        C0_l_calib, C0_u_calib, _, _ = model.predict_counterfactual_naive(alpha, X_calib_inter_1, Y0, Y1)

        coverage_0, coverage_1, interval_width_0, interval_width_1 = eval_po(Y1,Y0,C0_l,C0_u,C1_l,C1_u)

        if plot:
            # plot to check exchangeability of P(X,Y)
            for j in range(n_folds):
                X_calib_inter_0 = model.X_calib_inter_list[j][model.T_calib_inter_list[j]==0, :]
                Y_calib_inter_0 = model.Y_calib_inter_list[j][model.T_calib_inter_list[j]==0].reshape(-1,1)
                X_calib_inter_1 = model.X_calib_inter_list[j][model.T_calib_inter_list[j]==1, :]
                Y_calib_inter_1 = model.Y_calib_inter_list[j][model.T_calib_inter_list[j]==1].reshape(-1,1)

                if dr_use_Y == 1:
                    D_calib_inter_0 = np.concatenate([X_calib_inter_0, Y_calib_inter_0],axis=1)
                    D_calib_inter_1 = np.concatenate([X_calib_inter_1, Y_calib_inter_1],axis=1)
                elif dr_use_Y == 0:
                    D_calib_inter_0 = X_calib_inter_0
                    D_calib_inter_1 = X_calib_inter_1

                plot_tsne(D_calib_inter_0, D_test, j, T=0, dataset=dataset, fig_name="xydist_{dr_use_Y}")
                plot_tsne(D_calib_inter_1, D_test, j, T=1, dataset=dataset, fig_name="xydist_{dr_use_Y}")

    elif cf_method == 'inexact':
        model = SplitCP(data_obs=train_data,
                    data_inter=df_i,
                    n_folds=n_folds,
                    alpha=alpha, 
                    base_learner="RF", 
                    quantile_regression=quantile_regression) 
        
        C0_l_model, C0_u_model, C1_l_model, C1_u_model = model.predict_counterfactual_inexact(alpha, X_test, Y0, Y1, dr_use_Y=dr_use_Y)
        C0_l, C0_u, C1_l, C1_u = C0_l_model.predict(X_test), C0_u_model.predict(X_test), C1_l_model.predict(X_test), C1_u_model.predict(X_test)

        C1_l_calib, C1_u_calib = C1_l_model.predict(X_calib_inter_0), C1_u_model.predict(X_calib_inter_0)
        C0_l_calib, C0_u_calib = C0_l_model.predict(X_calib_inter_1), C0_u_model.predict(X_calib_inter_1)

        coverage_0, coverage_1, interval_width_0, interval_width_1 = eval_po(Y1,Y0,C0_l,C0_u,C1_l,C1_u)

    elif cf_method == 'exact':
        model = SplitCP(data_obs=train_data,
            data_inter=df_i,
            n_folds=n_folds,
            alpha=alpha / 2, 
            base_learner="RF", 
            quantile_regression=quantile_regression)
        
        C0_l_model, C0_u_model, C1_l_model, C1_u_model  = model.predict_counterfactual_exact(alpha / 2, X_test, Y0, Y1, dr_use_Y=dr_use_Y)
        C0_l, C0_u, C1_l, C1_u = C0_l_model.predict(X_test), C0_u_model.predict(X_test), C1_l_model.predict(X_test), C1_u_model.predict(X_test)
        
        C1_l_calib, C1_u_calib = C1_l_model.predict(X_calib_inter_0), C1_u_model.predict(X_calib_inter_0)
        C0_l_calib, C0_u_calib = C0_l_model.predict(X_calib_inter_1), C0_u_model.predict(X_calib_inter_1)

        coverage_0, coverage_1, interval_width_0, interval_width_1 = eval_po(Y1,Y0,C0_l,C0_u,C1_l,C1_u)

    
    elif cf_method == 'tcp':
        model = TCP(data_obs=train_data,
            data_inter=df_i,
            n_folds=n_folds,
            alpha=alpha / 2, 
            base_learner=base_learner,
            quantile_regression=quantile_regression,
            density_ratio_model=density_ratio_model,
            n_estimators=n_estimators,
            K=K)
        
        # alpha
        C0_l, C0_u = model.predict_counterfactual(X_test, 0)
        C1_l, C1_u = model.predict_counterfactual(X_test, 1)

        # This is for ite
        # for tcp ite

        # C1_l_calib, C1_u_calib = model.predict_counterfactual(X_calib_inter_0, 1)
        # C0_l_calib, C0_u_calib = model.predict_counterfactual(X_calib_inter_1, 0)

        C1_l_calib, C1_u_calib = None, None
        C0_l_calib, C0_u_calib = None, None

        coverage_0, coverage_1, interval_width_0, interval_width_1 = eval_po(Y1,Y0,C0_l,C0_u,C1_l,C1_u)

    # we consider all 3 ite evaluation methods
    res_list = []
    if cf_method != "tcp":
        ite_methods = ["naive", "inexact", "exact"]
    else:
        ite_methods = ["naive"]

    for ite_method in ite_methods:
        print(f'ite_method = {ite_method}')
        res = {}
        res['cf_method'] = cf_method
        res['ite_method'] = ite_method
        res['coverage_0'] = coverage_0
        res['coverage_1'] = coverage_1
        res['interval_width_0'] = interval_width_0
        res['interval_width_1'] = interval_width_1

        CI_ITE_l, CI_ITE_u = predict_ITE(alpha, X_test, C0_l, C0_u, C1_l, C1_u, 
                                         X_calib_inter_0, X_calib_inter_1,   
                                        C0_l_calib, C0_u_calib, C1_l_calib, C1_u_calib, 
                                        Y_calib_inter_0, Y_calib_inter_1, ite_method)
        
        coverage_ITE = np.mean((ITE_test >= CI_ITE_l) & (ITE_test <= CI_ITE_u))
        interval_width_ITE = np.mean(np.abs(CI_ITE_u - CI_ITE_l))
        print('Coverage of ITE', coverage_ITE)
        print('Interval width of ITE', interval_width_ITE)

        res['coverage_ITE'] = coverage_ITE
        res['interval_width'] = interval_width_ITE

        res_list.append(res)

    return res_list
    

def predict_ITE(alpha, X_test, C0_l, C0_u, C1_l, C1_u, 
                X_calib_inter_0, X_calib_inter_1, 
                C0_l_calib, C0_u_calib, C1_l_calib, C1_u_calib,
                Y_calib_inter_0, Y_calib_inter_1,
                ite_method):
    if ite_method == 'naive':
        CI_ITE_l = C1_l - C0_u
        CI_ITE_u = C1_u - C0_l

    elif ite_method == 'inexact':
        ITE_l_model = GradientBoostingRegressor()
        ITE_u_model = GradientBoostingRegressor()

        ite_calib_u =  np.concatenate(((C1_u_calib - Y_calib_inter_0), (Y_calib_inter_1 - C0_l_calib)), axis=0)
        ite_calib_l =  np.concatenate(((C1_l_calib - Y_calib_inter_0), (Y_calib_inter_1 - C0_u_calib)), axis=0)
        X_calib = np.concatenate((X_calib_inter_0, X_calib_inter_1), axis=0)
        ITE_l_model.fit(X_calib, ite_calib_l)
        ITE_u_model.fit(X_calib, ite_calib_u)

        CI_ITE_l = ITE_l_model.predict(X_test)
        CI_ITE_u = ITE_u_model.predict(X_test)

    elif ite_method == 'exact':
        ITE_l_model = GradientBoostingRegressor()
        ITE_u_model = GradientBoostingRegressor()

        ite_calib_u =  np.concatenate(((C1_u_calib - Y_calib_inter_0), (Y_calib_inter_1 - C0_l_calib)), axis=0)
        ite_calib_l =  np.concatenate(((C1_l_calib - Y_calib_inter_0), (Y_calib_inter_1 - C0_u_calib)), axis=0)
        X_calib = np.concatenate((X_calib_inter_0, X_calib_inter_1), axis=0)
        Y_calib = np.concatenate((Y_calib_inter_0, Y_calib_inter_1), axis=0)

        X_calib_train, X_calib_calib = train_test_split(X_calib, test_size=0.25, random_state=42)
        ite_calib_l_train, ite_calib_l_calib = train_test_split(ite_calib_l, test_size=0.25, random_state=42)
        ite_calib_u_train, ite_calib_u_calib = train_test_split(ite_calib_u, test_size=0.25, random_state=42)

        ITE_l_model.fit(X_calib_train, ite_calib_l_train)
        ITE_u_model.fit(X_calib_train, ite_calib_u_train)

        scores_l = np.abs(ITE_l_model.predict(X_calib_calib) - ite_calib_l_calib)
        offset_l = utils.standard_conformal(alpha, scores_l)
        scores_u = np.abs(ITE_u_model.predict(X_calib_calib) - ite_calib_u_calib)
        offset_u = utils.standard_conformal(alpha, scores_u)

        CI_ITE_l = ITE_l_model.predict(X_test) - offset_l
        CI_ITE_u = ITE_u_model.predict(X_test) + offset_u
    return CI_ITE_l, CI_ITE_u