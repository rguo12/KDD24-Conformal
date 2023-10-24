import numpy as np
import jax.numpy as jnp
import jax
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



def weighted_conformal_prediction(df_o, quantile_regression, alpha, test_frac, method):
       
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

    if method == 'counterfactual':
        model = WCP(data_obs=train_data,
                    alpha=alpha, 
                    base_learner="QRF", 
                    quantile_regression=quantile_regression) 
        model.fit()
        model.conformalize(alpha, method='naive')
        C0, C1 = model.predict_counterfactuals(alpha, X_test)

        coverage_0 = np.mean((Y0 >= C0[0]) & (Y0 <= C0[1]))
        coverage_1 = np.mean((Y1 >= C1[0]) & (Y1 <= C1[1]))
        interval_width_0 = jnp.mean(jnp.abs(C0[1] - C0[0]))
        interval_width_1 = jnp.mean(jnp.abs(C1[1] - C1[0]))
        return coverage_0, coverage_1, interval_width_0, interval_width_1
    
    elif method == 'naive_ITE':
        model = WCP(train_data=train_data,
                    alpha=alpha, 
                    base_learner="QRF", 
                    quantile_regression=quantile_regression) 
        model.fit()
        model.conformalize(alpha, method='naive')
    
        CI_ITE_l, CI_ITE_u = model.predict_ITE(alpha, X_test, method='naive')
        coverage_ITE = np.mean((ITE_test >= CI_ITE_l) & (ITE_test <= CI_ITE_u))
        interval_width_ITE = np.mean(np.abs(CI_ITE_u - CI_ITE_l))
        print('Coverage of ITE (naive)', coverage_ITE)
        print('Interval width of ITE (naive)', interval_width_ITE)

    elif method == 'nested_inexact':
        model = WCP(train_data=train_data,
                    alpha=alpha, 
                    base_learner="QRF", 
                    quantile_regression=quantile_regression) 
        model.fit()    
        model.conformalize(alpha, method='nested_inexact')
        CI_ITE_l, CI_ITE_u = model.predict_ITE(alpha, X_test, method='nested_inexact')
        coverage_ITE = np.mean((ITE_test >= CI_ITE_l) & (ITE_test <= CI_ITE_u))
        interval_width_ITE = np.mean(np.abs(CI_ITE_u - CI_ITE_l))
        print('Coverage of ITE (nested inexact)', coverage_ITE)
        print('Interval width of ITE (nested inexact)', interval_width_ITE)

    elif method == 'nested_exact':
        model = WCP(train_data=train_data,
                    alpha=alpha / 2, 
                    base_learner="QRF", 
                    quantile_regression=quantile_regression) 
        model.fit()
        model.conformalize(alpha / 2, method='nested_exact')
        CI_ITE_l, CI_ITE_u = model.predict_ITE(alpha / 2, X_test, method='nested_exact')
        coverage_ITE = np.mean((ITE_test >= CI_ITE_l) & (ITE_test <= CI_ITE_u))
        interval_width_ITE = np.mean(np.abs(CI_ITE_u - CI_ITE_l))
        print('Coverage of ITE (nested exact)', coverage_ITE)
        print('Interval width of ITE (nested exact)', interval_width_ITE)
    pause = True
    return 


def transductive_weighted_conformal(df_o, df_i, quantile_regression, alpha, test_frac, method):
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

    if method == 'counterfactual':
        model = TCP(data_obs=train_data,
                    data_inter=df_i,
                    alpha=alpha, 
                    base_learner="RF", 
                    quantile_regression=quantile_regression) 
        model.conformalize(alpha, method='naive')
        C0_l, C0_u, C1_l, C1_u = model.predict_counterfactuals(alpha, X_test, Y1, Y0)

        coverage_0 = jnp.mean((Y0 >= C0_l) & (Y0 <= C0_u))
        coverage_1 = jnp.mean((Y1 >= C1_l) & (Y1 <= C1_u))
        interval_width_0 = jnp.mean(jnp.abs(C0_u - C0_l))
        interval_width_1 = jnp.mean(jnp.abs(C1_u - C1_l))
        pause = True
        return coverage_0, coverage_1, interval_width_0, interval_width_1