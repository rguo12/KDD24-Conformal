import numpy as np
from scipy.stats import norm, beta
from scipy.special import erfinv, expit
import requests
import pyreadr

import pandas as pd
import os

TRAIN_DATASET = "./data/IHDP/ihdp_npci_1-100.train.npz"
TEST_DATASET = "./data/IHDP/ihdp_npci_1-100.test.npz"
TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"
PATH_dir = "./data/NLSM/data"

from pathlib import Path
from typing import Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, model_selection


def assemble_data(X, T, Y1, Y0, d, ps, mu0=None, mu1=None):
    Y = Y0.copy()
    Y[T] = Y1[T]
    data = np.column_stack((X, T, Y))
    column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
    df = pd.DataFrame(data, columns=column_names)
    df["ps"] = np.array(ps).reshape((-1,))
    df["Y1"] = Y1.reshape((-1,))
    df["Y0"] = Y0.reshape((-1,))
    df["CATE"] = Y1 - Y0

    if mu0 is not None and mu1 is not None:
        df["mu0"] = mu0 # Y0 w/o noise
        df["mu1"] = mu1

    return df


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


def generate_data(n_observation, n_intervention, d, gamma, alpha, confounding):
    
    def correlated_covariates(n, d):
        rho = 0.9
        X = np.random.normal(0., 1. , size=(n, d))
        fac = np.random.normal(0., 1. , size=(n, d))
        X = X * np.sqrt(1 - rho) + fac * np.sqrt(rho)
        return norm.cdf(X)
    
    # Generate observation data
    
    X = np.random.uniform(size=(n_observation, d)) # d-dimensional Uniform random variable X

    if confounding:
        # d-dimensional normal random variable U
        U = np.abs(np.random.normal(0., 1. , size=(n_observation, d)))
        
        # tau is a function of U
        tau = (2 / (1 + np.exp(-12 * ((U[:, 0] + U[:, 0]) / 2 - 0.5)))) * (2 / (1 + np.exp(-12 * ((U[:, 1] + U[:, 1]) / 2 - 0.5)))) 
        tau = tau.reshape((-1,))

        # tau_0 is a function of tau
        tau_0 = gamma * tau 
        # std = -np.log(np.minimum((U[:, 0] + U[:, 0]) / 2 + 1e-9, 0.99))
        std = 1.0
        ps = (1 + beta.cdf((U[:, 0] + U[:, 0]) / 2, 2, 4)) / 4

    else:
        tau = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))) 
        tau = tau.reshape((-1,))
        tau_0 = gamma * tau 
        std = -np.log(X[:, 0] + 1e-9)
        ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4

    errdist = np.random.normal(0., 1. , size=(n_observation, ))
    err_0 = np.random.normal(0., 1. , size=(n_observation, ))

    # Y0 = tau_0 + err_0 = gamma * tau_0 + err_0
    Y0 = tau_0 + 1 * err_0

    # Y1 = tau + sqrt(std) * errdist  
    Y1 = tau + np.sqrt(std) * errdist
    
    T = np.random.uniform(size=(n_observation, )) < ps
    Y = Y0.copy()
    Y[T] = Y1[T]
    
    data_observation = np.column_stack((X, T, Y))
    column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
    df_observation = pd.DataFrame(data_observation, columns=column_names)
    df_observation["ps"] = np.array(ps).reshape((-1,))
    df_observation["Y1"] = Y1.reshape((-1,))
    df_observation["Y0"] = Y0.reshape((-1,))
    df_observation["CATE"] = tau - tau_0
    df_observation["width"] = np.mean(np.sqrt(2)*(np.sqrt(2)*std) * erfinv(2*(1-(alpha/2))-1) * 2) 

    # Generate intervention data
    X = np.random.uniform(size=(n_intervention, d))

    if confounding:
        U = np.abs(np.random.normal(0., 1. , size=(n_intervention, d)))
        tau = (2 / (1 + np.exp(-12 * ((U[:, 0] + U[:, 0]) / 2 - 0.5)))) * (2 / (1 + np.exp(-12 * ((U[:, 1] + U[:, 1]) / 2 - 0.5)))) 
        tau = tau.reshape((-1,))
        tau_0 = gamma * tau 
        # std = -np.log(np.minimum((U[:, 0] + U[:, 0]) / 2 + 1e-9, 0.99))
        std = 1.0
        ps = (1 + beta.cdf((U[:, 0] + U[:, 0]) / 2, 2, 4)) / 4
    else:
        tau = (2 / (1 + np.exp(-12 * (X[:, 0] - 0.5)))) * (2 / (1 + np.exp(-12 * (X[:, 1] - 0.5)))) 
        tau = tau.reshape((-1,))
        tau_0 = gamma * tau 
        std = -np.log(X[:, 0] + 1e-9)
        ps = (1 + beta.cdf(X[:, 0], 2, 4)) / 4

    errdist = np.random.normal(0., 1. , size=(n_intervention, ))
    err_0 = np.random.normal(0., 1. , size=(n_intervention, ))
    
    Y0 = tau_0 + 1 * err_0  
    Y1 = tau + np.sqrt(std) * errdist
    ps = 0.5 * np.ones(shape=(n_intervention, ))
    T = np.random.uniform(size=(n_intervention, )) < ps
    Y = Y0.copy()
    Y[T] = Y1[T]
    
    data_intervention = np.column_stack((X, T, Y))
    column_names = [f'X{i}' for i in range(1, d+1)] + ['T', 'Y']
    df_intervention = pd.DataFrame(data_intervention, columns=column_names)
    df_intervention["ps"] = np.array(ps).reshape((-1,))
    df_intervention["Y1"] = Y1.reshape((-1,))
    df_intervention["Y0"] = Y0.reshape((-1,))
    df_intervention["CATE"] = tau - tau_0
    df_intervention["width"] = np.mean(np.sqrt(2)*(np.sqrt(2)*std)*erfinv(2*(1-(alpha/2))-1) * 2) 

    return df_observation, df_intervention




def IHDP_w_HC(n_intervention:int, seed:int, d:int=24,
              hidden_confounding=True, beta_u:float=0.5, root:str="./data/IHDP"):
    # adapted from https://github.com/anndvision/quince/blob/main/quince/library/datasets/ihdp.py

    """
    IHDP with Hidden Confounding

    Args:
        root (_type_): _description_
        split (_type_): _description_
        mode (_type_): _description_
        seed (_type_): _description_
        hidden_confounding (_type_): _description_
        beta_u (_type_, optional): strength of hidden confounding, from 0.1 to 0.5.

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    _CONTINUOUS_COVARIATES = [
        "bw",
        "b.head",
        "preterm",
        "birth.o",
        "nnhealth",
        "momage",
    ]

    _BINARY_COVARIATES = [
        "sex",
        "twin",
        "mom.lths",
        "mom.hs",
        "mom.scoll",
        "cig",
        "first",
        "booze",
        "drugs",
        "work.dur",
        "prenatal",
        "ark",
        "ein",
        "har",
        "mia",
        "pen",
        "tex",
        "was",
    ]

    _HIDDEN_COVARIATE = [
        "b.marr",
    ]

    # root = Path.home() / "quince_datasets" if root is None else Path(root)
    data_path = os.path.join(root,"ihdp.RData")
    # Download data if necessary
    if not os.path.isfile(data_path):
        # root.mkdir(parents=True, exist_ok=True)
        r = requests.get(
            "https://github.com/vdorie/npci/raw/master/examples/ihdp_sim/data/ihdp.RData"
        )
        with open(data_path, "wb") as f:
            f.write(r.content)
    df = pyreadr.read_r(data_path)["ihdp"]
    # Make observational as per Hill 2011
    df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
    df = df[
        _CONTINUOUS_COVARIATES + _BINARY_COVARIATES + _HIDDEN_COVARIATE + ["treat"]
    ]
    # Standardize continuous covariates
    df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
        df[_CONTINUOUS_COVARIATES]
    )
    
    # Generate response surfaces
    rng = np.random.default_rng(seed)
    x = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES]
    u = df[_HIDDEN_COVARIATE]
    t = df["treat"]

    # randomly select coefficients for x and u
    beta_x = rng.choice(
        [0.0, 0.1, 0.2, 0.3, 0.4], size=(24,), p=[0.6, 0.1, 0.1, 0.1, 0.1]
    )

    # beta_u = (
    #     rng.choice(
    #         [0.1, 0.2, 0.3, 0.4, 0.5], size=(1,), p=[0.2, 0.2, 0.2, 0.2, 0.2]
    #     )
    #     if beta_u is None
    #     else np.asarray([beta_u])
    # )

    beta_u = np.asarray([beta_u])

    # mu0 is exponential (harder)
    mu0 = np.exp((x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u))
    df["mu0"] = mu0
    
    # mu1 is linear
    mu1 = (x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u)
    omega = (mu1[t == 1] - mu0[t == 1]).mean(0) - 4
    mu1 -= omega
    df["mu1"] = mu1
    eps = rng.normal(size=t.shape)
    y0 = mu0 + eps
    df["y0"] = y0
    y1 = mu1 + eps
    df["y1"] = y1
    y = t * y1 + (1 - t) * y0
    df["y"] = y

    n = len(y)

    # obs int split
    df_obs_raw, df_int_raw = model_selection.train_test_split(
        df, test_size=float(n_intervention)/n, random_state=seed
    )

    covars = _CONTINUOUS_COVARIATES + _BINARY_COVARIATES
    covars = covars + _HIDDEN_COVARIATE if not hidden_confounding else covars
    
    U_obs = df_obs_raw[_HIDDEN_COVARIATE].to_numpy(dtype="float32")
    X_obs = df_obs_raw[covars].to_numpy(dtype="float32")
    T_obs = df_obs_raw["treat"].to_numpy(dtype="int")
    mu0_obs = df_obs_raw["mu0"].to_numpy(dtype="float32")
    mu1_obs = df_obs_raw["mu1"].to_numpy(dtype="float32")
    Y0_obs = df_obs_raw["y0"].to_numpy(dtype="float32")
    Y1_obs = df_obs_raw["y1"].to_numpy(dtype="float32")

    model = LogisticRegression()
    model.fit(X_obs, T_obs)
    ps_obs = model.predict_proba(X_obs)[:, 1]

    df_obs = assemble_data(X_obs, T_obs, Y1_obs, Y0_obs, d, ps_obs, mu1=mu1_obs, mu0=mu0_obs)

    U_int = df_int_raw[_HIDDEN_COVARIATE].to_numpy(dtype="float32")
    X_int = df_int_raw[covars].to_numpy(dtype="float32")
    # override the true treatment?
    ps_int = 0.5 * np.ones_like(U_int)
    T_int = (np.random.uniform(size=U_int.shape) < ps_int).astype("int") #df_obs_raw["treat"].to_numpy(dtype="float32")
    mu0_int = df_int_raw["mu0"].to_numpy(dtype="float32")
    mu1_int = df_int_raw["mu1"].to_numpy(dtype="float32")
    Y0_int = df_int_raw["y0"].to_numpy(dtype="float32")
    Y1_int = df_int_raw["y1"].to_numpy(dtype="float32")

    df_int = assemble_data(X_int, T_int, Y1_int, Y0_int, d, ps_int, mu1=mu1_int, mu0=mu0_int)

    return df_obs, df_int


def generate_cevae_data(n_observation, n_intervention, d=1, err_scale=0.1, ps_strength=0.6):

    def generate_confounder(n):
        return np.random.normal(0., 1., size=(n, ))

    def generate_covariate(U, sigma_z0=5.0, sigma_z1=3.0):
        variance_X = sigma_z0**2*(1-U) + sigma_z1**2*U
        mean_X = U
        return np.random.normal(0., 1., size=U.shape) * variance_X + mean_X

    def generate_treatment(U, ps_strength=0.6, intervention=False):

        ps = ps_strength*U + (1-ps_strength)*(1-U)

        if intervention:
            return np.random.uniform(size=U.shape) < 0.5 * np.ones_like(U), ps
        else:
            return np.random.uniform(size=U.shape) < ps, ps

    def generate_outcomes(U, n, err_scale=0.1):
        errY1 = np.random.normal(0., 1., size=(n, )) * err_scale
        errY0 = np.random.normal(0., 1., size=(n, )) * err_scale
        tau1 = expit(3.0*(U+2))
        tau0 = expit(3.0*(U-2))
        Y1 = tau1 + errY1
        Y0 = tau0 + errY0
        return Y1, Y0
    
    # Generate observation data
    U_obs = generate_confounder(n_observation)
    X_obs = generate_covariate(U_obs)
    T_obs, ps_obs = generate_treatment(U_obs, ps_strength)
    Y1_obs, Y0_obs = generate_outcomes(U_obs, n_observation, err_scale)
    df_observation = assemble_data(X_obs, T_obs, Y1_obs, Y0_obs, d, ps_obs)

    # Generate intervention data
    U_int = generate_confounder(n_intervention)
    X_int = generate_covariate(U_int)
    T_int, ps_int = generate_treatment(U_int, ps_strength, intervention=True) #ps is the true ps, just for recording, not 0.5
    Y1_int, Y0_int = generate_outcomes(U_int, n_intervention, err_scale)
    df_intervention = assemble_data(X_int, T_int, Y1_int, Y0_int, d, ps_int)

    return df_observation, df_intervention




def generate_leihua_li_data():

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