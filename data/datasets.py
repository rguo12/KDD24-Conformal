import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
from jax.scipy.stats import norm, beta
from jax.scipy.special import erfinv
import os

TRAIN_DATASET = "./data/IHDP/ihdp_npci_1-100.train.npz"
TEST_DATASET = "./data/IHDP/ihdp_npci_1-100.test.npz"
TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"
PATH_dir = "./data/NLSM/data"

from pathlib import Path
from typing import Any, Tuple



def convert(npz_file, scale=None):

    npz_data = np.load(npz_file)
    scales = []
    
    x = npz_data['x']
    t = npz_data['t']
    yf = npz_data['yf']
    ycf = npz_data['ycf']
    mu0 = npz_data['mu0']
    mu1 = npz_data['mu1']
    
    num_realizations = x.shape[2]
    
    dataframes = []
    
    for i in range(num_realizations):
        
        x_realization = x[:, :, i]
        t_realization = t[:, i]
        yf_realization = yf[:, i]
        ycf_realization = ycf[:, i]
        mu1_realization = mu1[:, i]
        mu0_realization = mu0[:, i]

        model = LogisticRegression()
        model.fit(x_realization, t_realization) 
        
        df = pd.DataFrame(x_realization, columns=[f'X{j + 1}' for j in range(x_realization.shape[1])])
        df['T'] = t_realization
        df['Y'] = yf_realization
        df['Y_cf'] = ycf_realization
        df['Y1'] = yf_realization * t_realization + ycf_realization * (1 - t_realization)  #mu1_realization
        df['Y0'] = ycf_realization * t_realization + yf_realization * (1 - t_realization)#mu0_realization
        df['ITE'] = df['Y1'] - df['Y0']
        df["ps"] = model.predict_proba(x_realization)[:, 1]

        df["CATE"] = mu1_realization - mu0_realization

        sd_cate = np.sqrt((np.array(df["CATE"])).var())

        if scale is None:
            
            if sd_cate > 1:

                error_0 = np.array(df['Y0']) - mu0_realization 
                error_1 = np.array(df['Y1']) - mu1_realization

                mu0_ = mu0_realization / sd_cate
                mu1_ = mu1_realization / sd_cate

                scales.append(sd_cate) 
              
                df['Y0'] = mu0_ + error_0
                df['Y1'] = mu1_ + error_1
                df['ITE'] = df['Y1'] - df['Y0']
                df["CATE"] = mu1_ - mu0_ 
          
            else:

                scales.append(1)

        elif scale is not None:
            
            # test data
            error_0 = np.array(df['Y0']) - mu0_realization 
            error_1 = np.array(df['Y1']) - mu1_realization

            mu0_ = mu0_realization / scale[i]
            mu1_ = mu1_realization / scale[i]

            df['Y0'] = mu0_ + error_0
            df['Y1'] = mu1_ + error_1
            df['ITE'] = df['Y1'] - df['Y0']
            df["CATE"] = mu1_ - mu0_ 
        
        dataframes.append(df)
    
    return dataframes, scales


def IHDP_data():
    
    train = './data/IHDP/ihdp_npci_1-100.train.npz'
    test = './data/IHDP/ihdp_npci_1-100.test.npz'
    
    train_data, scale = convert(train)
    test_data, _ = convert(test, scale)
    
    return train_data, test_data


def NLSM_data():

    NLSM_files = os.listdir(PATH_dir) 
    
    dataset = []

    for nlsm_file in NLSM_files: 
        
        df = pd.read_csv(PATH_dir + "/" + nlsm_file)
        df["CATE"] = df["Etau"]

        dataset.append(df)

    return dataset


def generate_data(rng_key, n_observation, n_intervention, d, gamma, alpha, confouding):
    
    def correlated_covariates(rng_key, n, d):
        rho = 0.9
        rng_key, _ = jax.random.split(rng_key)
        X = jax.random.normal(rng_key, shape=(n, d))
        fac = jax.random.normal(rng_key, shape=(n, d))
        X = X * jnp.sqrt(1 - rho) + fac * jnp.sqrt(rho)
        return norm.cdf(X)
    
    # Generate observation data
    rng_key, _ = jax.random.split(rng_key)
    X = jax.random.uniform(rng_key, shape=(n_observation, d))
    if confouding:
        rng_key, _ = jax.random.split(rng_key)
        U = jnp.abs(jax.random.normal(rng_key, shape=(n_observation, d)))
        tau = (2 / (1 + jnp.exp(-12 * ((U[:, 0] + U[:, 0]) / 2 - 0.5)))) * (2 / (1 + jnp.exp(-12 * ((U[:, 1] + U[:, 1]) / 2 - 0.5)))) 
        tau = tau.reshape((-1,))
        tau_0 = gamma * tau 
        # std = -jnp.log(jnp.minimum((U[:, 0] + U[:, 0]) / 2 + 1e-9, 0.99))
        std = 1.0
        ps = (1 + beta.cdf((U[:, 0] + U[:, 0]) / 2, 2, 4)) / 4
    else:
        tau = (2 / (1 + jnp.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + jnp.exp(-12 * (X[:, 1] - 0.5)))) 
        tau = tau.reshape((-1,))
        tau_0 = gamma * tau 
        std = -jnp.log(X[:, 0] + 1e-9)
        ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4

    rng_key, _ = jax.random.split(rng_key)
    errdist = jax.random.normal(rng_key, shape=(n_observation, ))
    rng_key, _ = jax.random.split(rng_key)
    err_0 = jax.random.normal(rng_key, shape=(n_observation, ))

    Y0 = tau_0 + 1 * err_0  
    Y1 = tau + jnp.sqrt(std) * errdist
    rng_key , _ = jax.random.split(rng_key)
    T = jax.random.uniform(rng_key, shape=(n_observation, )) < ps
    Y = Y0.copy()
    Y = Y.at[T].set(Y1[T])
    
    data_observation = jnp.column_stack((X, T, Y))
    column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
    df_observation = pd.DataFrame(data_observation, columns=column_names)
    df_observation["ps"] = jnp.array(ps).reshape((-1,))
    df_observation["Y1"] = Y1.reshape((-1,))
    df_observation["Y0"] = Y0.reshape((-1,))
    df_observation["CATE"] = tau - tau_0
    df_observation["width"] = jnp.mean(jnp.sqrt(2)*(jnp.sqrt(2)*std) * erfinv(2*(1-(alpha/2))-1) * 2) 

    # Generate intervention data
    rng_key, _ = jax.random.split(rng_key)
    X = jax.random.uniform(rng_key, shape=(n_intervention, d))

    if confouding:
        rng_key, _ = jax.random.split(rng_key)
        U = jnp.abs(jax.random.normal(rng_key, shape=(n_intervention, d)))
        tau = (2 / (1 + jnp.exp(-12 * ((U[:, 0] + U[:, 0]) / 2 - 0.5)))) * (2 / (1 + jnp.exp(-12 * ((U[:, 1] + U[:, 1]) / 2 - 0.5)))) 
        tau = tau.reshape((-1,))
        tau_0 = gamma * tau 
        # std = -jnp.log(jnp.minimum((U[:, 0] + U[:, 0]) / 2 + 1e-9, 0.99))
        std = 1.0
        ps = (1 + beta.cdf((U[:, 0] + U[:, 0]) / 2, 2, 4)) / 4
    else:
        tau = (2 / (1 + jnp.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + jnp.exp(-12 * (X[:, 1] - 0.5)))) 
        tau = tau.reshape((-1,))
        tau_0 = gamma * tau 
        std = -jnp.log(X[:, 0] + 1e-9)
        ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4

    rng_key, _ = jax.random.split(rng_key)
    errdist = jax.random.normal(rng_key, shape=(n_intervention, ))
    rng_key, _ = jax.random.split(rng_key)
    err_0 = jax.random.normal(rng_key, shape=(n_intervention, ))
    
    Y0 = tau_0 + 1 * err_0  
    Y1 = tau + jnp.sqrt(std) * errdist
    rng_key , _ = jax.random.split(rng_key)
    T = jax.random.randint(rng_key, shape=(n_intervention, ), minval=0, maxval=2)
    Y = Y0.copy()
    Y = Y.at[T].set(Y1[T])
    
    data_intervention = jnp.column_stack((X, T, Y))
    column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
    df_intervention = pd.DataFrame(data_intervention, columns=column_names)
    df_intervention["ps"] = jnp.array(ps).reshape((-1,))
    df_intervention["Y1"] = Y1.reshape((-1,))
    df_intervention["Y0"] = Y0.reshape((-1,))
    df_intervention["CATE"] = tau - tau_0
    df_intervention["width"] = jnp.mean(jnp.sqrt(2)*(jnp.sqrt(2)*std)*erfinv(2*(1-(alpha/2))-1) * 2) 

    return df_observation, df_intervention


def generate_lilei_hua_data():
    # Replicate the demo from https://lihualei71.github.io/cfcausal/articles/cfcausal_demo.html
    np.random.seed(2020)

    def genY(X):
        n = X.shape[0]
        term1 = 2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))
        term2 = 2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))
        random_noise = norm.rvs(size=n)  # rnorm in R
        return term1 * term2 + random_noise

    n = 2000
    d = 10

    # Generate random uniform data
    X = np.random.rand(n, d)

    # Apply genY function
    Y1 = genY(X)
    Y0 = norm.rvs(size=n)  # rnorm in R

    # Calculate 'ps' and 'T', then determine 'Y' based on 'T'
    ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4
    T = (ps < np.random.rand(n)).astype(int)
    Y = np.where(T == 1, Y1, Y0)  # ifelse in R

    # Generate testing data
    ntest = 1000
    Xtest = np.random.rand(ntest, d)

    # Calculate 'pstest' and 'Ttest', then determine 'Ytest' based on 'Ttest'
    pstest = (1 + beta.cdf(Xtest[:, 0], 2, 4)) / 4
    Ttest = (pstest < np.random.rand(ntest)).astype(int)
    Y1test = genY(Xtest)
    Y0test = norm.rvs(size=ntest)  # rnorm in R
    Ytest = np.where(Ttest == 1, Y1test, Y0test)  # ifelse in R


    data_train = np.column_stack((X, T, Y))
    column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
    df_train = pd.DataFrame(data_train, columns=column_names)
    df_train["ps"] = np.array(ps).reshape((-1,))
    df_train["Y1"] = Y1
    df_train["Y0"] = Y0

    data_test = np.column_stack((Xtest, Ttest, Ytest))
    column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
    df_test = pd.DataFrame(data_test, columns=column_names)
    df_test["ps"] = np.array(pstest).reshape((-1,))
    df_test["Y1"] = Y1test
    df_test["Y0"] = Y0test

    return df_train, df_test