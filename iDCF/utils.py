import bottleneck as bn
import numpy as np
import random
import os
import ray
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

base_learners_dict = dict({"GBM": GradientBoostingRegressor, 
                           "RF": RandomForestQuantileRegressor})



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

class PinballLoss(nn.Module):
    def __init__(self, quantile):
        """
        Initialize the Pinball loss function.

        Parameters:
        quantile (float): The quantile to be estimated, e.g., 0.1 for 10th percentile.
        """
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, predictions, targets):
        """
        Forward pass of the Pinball loss function.

        Parameters:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): True values.
        
        Returns:
        torch.Tensor: Computed Pinball loss.
        """
        errors = targets - predictions
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)




def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_index(data_size, split_ratio, random_split, seed=1234):
    all_user_id = np.arange(data_size)
    np.random.seed(seed)
    if random_split:
        np.random.shuffle(all_user_id)
    validation_index = all_user_id[:int(data_size * split_ratio)]
    test_index = all_user_id[int(data_size * split_ratio):]
    return validation_index, test_index


def split_by_item(df, ratio, seed=1234):
    data_group_by_user = df.groupby("user_id")
    train_list, test_list = list(), list()
    np.random.seed(seed)
    for i, (_, group) in enumerate(data_group_by_user):
        n_items = len(group)
        sampled_idx = np.zeros(n_items, dtype="bool")
        sampled_idx[np.random.choice(n_items, size=int(ratio * n_items), replace=False)] = True
        train_list.append(group[np.logical_not(sampled_idx)])
        test_list.append(group[sampled_idx])
    data_train = pd.concat(train_list)
    data_test = pd.concat(test_list)
    return data_train, data_test


def split_by_user(df, ratio, seed=1234):
    np.random.seed(seed)
    unique_uids = df["user_id"].unique()
    test_users = np.random.choice(unique_uids, size=int(unique_uids.size * ratio), replace=False)
    val_users = np.setdiff1d(unique_uids, test_users)
    df_val = df.loc[df["user_id"].isin(val_users)]
    df_test = df.loc[df["user_id"].isin(test_users)]
    return df_val, df_test, val_users, test_users



def split_random(df, tr_ratio, ts_ratio, seed=42):
    """
    Split the dataframe into training, validation, and test sets.

    :param df: DataFrame to split.
    :param tr_ratio: Ratio for training set.
    :param ts_ratio: Ratio for test set.
    :return: Tuple of DataFrames (df_train, df_val, df_test).
    """
    # Ensure the ratios sum up to 1 or less
    if tr_ratio + ts_ratio >= 1:
        raise ValueError("Sum of training and validation ratios should be less than 1.")

    # Splitting into training and temp (val + test) sets
    df_train, df_temp = train_test_split(df, train_size=tr_ratio, random_state=seed)

    # Adjusting validation ratio for splitting temp into val and test
    adjusted_ts_ratio = ts_ratio / (1 - tr_ratio)
    adjusted_val_ratio = 1-adjusted_ts_ratio

    # Splitting temp into validation and test sets
    df_val, df_test = train_test_split(df_temp, train_size=adjusted_val_ratio, random_state=seed)

    return df_train, df_val, df_test


def split2_random(df, ts_ratio, seed=42):
    """
    Split the dataframe into training, validation, and test sets.

    :param df: DataFrame to split.
    :param tr_ratio: Ratio for training set.
    :param ts_ratio: Ratio for test set.
    :return: Tuple of DataFrames (df_train, df_val, df_test).
    """
    # Ensure the ratios sum up to 1 or less
    if ts_ratio >= 1:
        raise ValueError("Sum of training and validation ratios should be less than 1.")

    tr_ratio = 1-ts_ratio
    # Splitting into training and temp (val + test) sets
    df_train, df_test = train_test_split(df, train_size=tr_ratio, random_state=seed)

    return df_train, df_test

def df_to_csr(df, shape):
    rows = df["user_id"]
    cols = df["item_id"]
    values = df["rating"]
    mat = csr_matrix((values, (rows, cols)))
    # mat = mat[mat.getnnz(axis=1) > 0]
    # assert mat.shape == shape
    return mat


def np_to_csr(array):
    rows = array[:, 0].astype(int)
    cols = array[:, 1].astype(int)
    values = array[:, 2]
    mat = csr_matrix((values, (rows, cols)))
    return mat


def construct_rating_dataset(train_df_path, random_df_path, test_ratio, split_index=False):
    train_df = pd.read_csv(train_df_path)
    # train_df = train_df.loc[train_df["user_id"] < 5400]
    random_df = pd.read_csv(random_df_path)

    # val_df, test_df = split_by_item(random_df, validation_ratio)
    val_df, test_df, val_users, test_users = split_by_user(random_df, test_ratio)
    if split_index:
        return train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy(), val_users, test_users
    else:
        return train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy()


def construct_rating_dataset_for_naive(random_df_path, train_ratio, test_ratio, split_index=False):
    """
    Naive method only uses random (interventional) data
    """
    # train_df = pd.read_csv(train_df_path)
    # train_df = train_df.loc[train_df["user_id"] < 5400]
    random_df = pd.read_csv(random_df_path)

    # val_df, test_df = split_by_item(random_df, validation_ratio)
    train_df, val_df, test_df = split_random(random_df, train_ratio, test_ratio, seed=42)

    return train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy()

def construct_vae_dataset(df_path, train_ratio, split_test=False, test_test_ratio=0.5, seed=1234):
    df = pd.read_csv(df_path)
    unique_users = df["user_id"].unique()

    n_users = unique_users.shape[0]
    n_items = df["item_id"].max() + 1
    if train_ratio == 1:
        return df_to_csr(df, shape=(n_users, n_items)).toarray()
    n_train_users = int(train_ratio * n_users)

    np.random.seed(seed)
    train_user_index = np.random.choice(unique_users, size=n_train_users, replace=False)
    train_user_index = np.sort(train_user_index)
    test_user_index = np.setdiff1d(unique_users, train_user_index)

    if split_test:
        pass
        # index = df["user_id"].isin(train_user_index)
        # train_df = df.loc[index]
        # test_df = df.loc[~index]
        # test_train, test_test = split_by_item(test_df, test_test_ratio)
        # train_matrix = df_to_csr(train_df, shape=(n_train_users, n_items))
        # test_tr_matrix = df_to_csr(test_train, shape=(n_users - n_train_users, n_items))
        # test_te_matrix = df_to_csr(test_test, shape=(n_users - n_train_users, n_items))
        # return train_matrix.toarray(), test_tr_matrix.toarray(), test_te_matrix.toarray(), train_user_index
    else:
        matrix = df_to_csr(df, (n_users, n_items))
        train_matrix = matrix[train_user_index]
        test_matrix = matrix[test_user_index]
        return train_matrix.toarray(), test_matrix.toarray(), train_user_index, test_user_index


def load_coat_by_ui_pair(path="data_process/coat/", validation_ratio=0.3):
    train_data_raw = pd.read_table(path + "train.ascii").to_numpy()
    test_data_raw = pd.read_table(path + "test.ascii").to_numpy()
    user_feature = pd.read_table(path + "user_item_features/user_features.ascii", sep=" ", header=None).to_numpy()

    val_data = np.zeros_like(test_data_raw)
    test_data = np.zeros_like(test_data_raw)
    for i, row in enumerate(test_data_raw):
        nonzero_items = row.nonzero()[0]
        val_iid = np.random.choice(nonzero_items, size=int(len(nonzero_items) * validation_ratio), replace=False)
        test_iid = np.setdiff1d(nonzero_items, val_iid)
        val_data[i][val_iid] = test_data_raw[val_iid]
        test_data[i][test_iid] = test_data_raw[test_iid]
    train_matrix = csr_matrix(train_data_raw)
    val_matrix = csr_matrix(val_data)
    test_matrix = csr_matrix(test_data)
    return train_matrix, val_matrix, test_matrix, user_feature


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)])
    valid_index = np.nonzero(IDCG)
    return DCG[valid_index] / IDCG[valid_index]


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = heldout_batch > 0
    hit = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    total_size = X_true_binary.sum(axis=1)
    valid_index = np.nonzero(total_size)
    recall = hit[valid_index] / total_size[valid_index]
    return recall


@ray.remote
def NDCG_RECALL_at_k_batch_parallel(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)])
    valid_index = np.nonzero(IDCG)
    ndcg = DCG[valid_index] / IDCG[valid_index]

    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]] = True

    X_true_binary = heldout_batch > 0
    hit = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    total_size = X_true_binary.sum(axis=1)
    valid_index = np.nonzero(total_size)
    recall = hit[valid_index] / total_size[valid_index]
    return np.concatenate((ndcg.reshape(-1, 1), recall.reshape(-1, 1)), axis=1)


def cal_ndcg_recall_parallel(num_workers, X_pred, heldout_batch, k=100):
    prediction = X_pred
    labels = heldout_batch
    lens = X_pred.shape[0]
    piece_lens = int(lens / num_workers)
    task = []
    rounds = num_workers if lens % num_workers == 0 else num_workers + 1
    for i in range(rounds):
        start = i * piece_lens
        end = min((i + 1) * piece_lens, lens)
        x = prediction[start:end]
        y = labels[start:end]
        task.append(NDCG_RECALL_at_k_batch_parallel.remote(x, y, k))
    res = ray.get(task)
    return np.concatenate(res, axis=0)

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

def get_density_ratio_data(data_loader, model, device="cpu"):
    # collect data
    labels = []
    embeddings = []
    for index, (uid, iid, rating) in enumerate(data_loader):
        uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
        U, I = model.get_embedding(uid,iid)
        ui_embedding = torch.cat([U,I],dim=1).detach().cpu().numpy()
        embeddings.append(ui_embedding)
        labels.extend(rating.tolist())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels).reshape(-1,1)

    D = np.concatenate([embeddings,labels],axis=1)
    return D

def train_density_ratio(train_obs_loader, train_int_loader, model_u, model_l, device="cpu", which_model="u",dr_model="DR"):

    if which_model == "u":
        model = model_u
    else:
        model = model_l

    D_train_obs = get_density_ratio_data(train_obs_loader, model, device=device)
    D_train_int = get_density_ratio_data(train_int_loader, model, device=device)
    
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

def mf_conf_eval_splitcp(cal_obs_loaders:list, cal_int_loaders:list, test_int_loaders:list, 
                         model_u_list:list, model_l_list:list, dr_model_list:list,
                        device:str="cpu", params=None, alpha:float=0.1, standardize:bool=True,
                        base_learner:str="RF", n_estimators:int=10, exact:bool=False,
                        dr_model:str="DR"):
    
    print("start conf eval for our split cp methods")

    # first do weighted CP on calib

    n_folds = len(cal_obs_loaders)

    coverages = []
    interval_widths = []
    
    for i in range(n_folds):
        model_u = model_u_list[i]
        model_l = model_l_list[i]
        density_model = dr_model_list[i]

        cal_obs_loader = cal_obs_loaders[i]
        cal_int_loader = cal_int_loaders[i]
        test_int_loader = test_int_loaders[i]

        model_u.eval()
        model_l.eval()
        
        # compute scores on cal_obs
        scores_list = mf_calib(cal_obs_loader, model_u, model_l, 
                               device=device, alpha=alpha, params=params, standardize=standardize)

        D_calib_obs = get_density_ratio_data(cal_obs_loader, model_u, device=device)
        D_calib_int = get_density_ratio_data(cal_int_loader, model_u, device=device)
        D_test_int = get_density_ratio_data(test_int_loader, model_u, device=device)

        X_calib_int = D_calib_int[:,:-1] # drop the last dimension
        X_test_int = D_test_int[:,:-1]

        y_calib_int =  D_calib_int[:,-1]
        y_test_int =  D_test_int[:,-1]

        if exact:
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

        else:
            if dr_model == "DR":
                weights_calib_obs = density_model.compute_density_ratio(D_calib_obs)
                weights_calib_int = density_model.compute_density_ratio(D_calib_int)
            
            elif dr_model == "MLP":
                p_calib_obs = density_model.predict_proba(D_calib_obs)[:,1]
                weights_calib_obs = (1. - p_calib_obs) / p_calib_obs #TODO: double check

                p_calib_int = density_model.predict_proba(D_calib_int)[:,1]
                weights_calib_int = (1. - p_calib_int) / p_calib_int

        print(f"weight obs: mean {np.mean(weights_calib_obs)}, std {np.std(weights_calib_obs)}")
        print(f"weight int: mean {np.mean(weights_calib_int)}, std {np.std(weights_calib_int)}")

        # weights are different for exact and inexact
        offset = weighted_conformal(alpha, weights_calib_obs, weights_calib_int, scores_list)[0]
        print(f"offset: {offset}")

        # compute predicted y_u, y_l on calib_int data
        with torch.no_grad():
            labels, y_u_list, y_l_list = list(), list(), list()
            for index, (uid, iid, rating) in enumerate(cal_int_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                y_u = model_u.predict(uid, iid)
                y_l = model_l.predict(uid, iid)

                if standardize:
                    y_u = params["min_val"] + y_u * (params["max_val"] - params["min_val"])
                    y_l = params["min_val"] + y_l * (params["max_val"] - params["min_val"])
                
                labels.extend(rating.tolist())
                y_u_list.append(y_u.detach().cpu().numpy())
                y_l_list.append(y_l.detach().cpu().numpy())
                
            y_u_list = np.concatenate(y_u_list)
            y_l_list = np.concatenate(y_l_list)

            y_u_list = y_u_list + offset
            y_l_list = y_l_list - offset

            # inexact method: fit models to predict y_u, y_l from features

            print("fit models for inexact upper/lower bound prediction")
            # not quantile regression
            y_u_predictor = RandomForestRegressor(n_estimators=n_estimators) #base_learners_dict[base_learner](**first_CQR_args_u)
            y_l_predictor = RandomForestRegressor(n_estimators=n_estimators) #base_learners_dict[base_learner](**first_CQR_args_l)

            if exact:
                y_u_fold_one = y_u_list[:n_calib_int_fold_one]
                y_l_fold_one = y_l_list[:n_calib_int_fold_one]
                # y_u_fold_two = y_u_list[n_calib_int_fold_one:]
                # y_l_fold_two = y_l_list[n_calib_int_fold_one:]

                y_u_predictor.fit(X_calib_int_fold_one, y_u_fold_one)
                y_l_predictor.fit(X_calib_int_fold_one, y_l_fold_one)

                y_u_hat_calib_int_fold_two = y_u_predictor.predict(X_calib_int_fold_two)
                y_l_hat_calib_int_fold_two = y_l_predictor.predict(X_calib_int_fold_two)

                # run conf the 2nd time, using calib_int fold one as training and fold two as calibration
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

            labels = np.array(labels)
            coverage = np.mean((labels >= predicts_l_test_int) & (labels <= predicts_u_test_int))
            interval_width = np.mean(np.abs(predicts_u_test_int - predicts_l_test_int))

            coverages.append(coverage)
            interval_widths.append(interval_width)
                
    return coverages, interval_widths


def mf_conf_eval_naive(cal_loaders:list, test_loaders:list, model_u_list:list, model_l_list:list, 
                 device="cpu", params=None, alpha=0.1, standardize=True):

    n_folds = len(cal_loaders)

    # offset_list = []
    # predict_u_list = []
    # predict_l_list = []

    coverages = []
    interval_widths = []

    for i in range(n_folds):
        model_u = model_u_list[i]
        model_l = model_l_list[i]
        cal_loader = cal_loaders[i]
        test_loader = test_loaders[i]

        model_u.eval()
        model_l.eval()
        
        scores_list = mf_calib(cal_loader, model_u, model_l, device=device, alpha=alpha, standardize=standardize)
        offset = standard_conformal(alpha, scores_list)

        with torch.no_grad():
            labels, predicts_u, predicts_l = list(), list(), list()
            for index, (uid, iid, rating) in enumerate(test_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict_u = model_u.predict(uid, iid)
                predict_l = model_l.predict(uid, iid)
                if standardize:
                    predict_u = params["min_val"] + predict_u * (params["max_val"] - params["min_val"])
                    predict_l = params["min_val"] + predict_l * (params["max_val"] - params["min_val"])
                
                labels.extend(rating.tolist())
                predicts_u.append(predict_u.detach().cpu().numpy())
                predicts_l.append(predict_l.detach().cpu().numpy())
                
            predicts_u = np.concatenate(predicts_u)
            predicts_l = np.concatenate(predicts_l)

            predicts_u = predicts_u + offset
            predicts_l = predicts_l - offset

            # predict_u_list.append(predicts_u)
            # predict_l_list.append(predicts_l)

            labels = np.array(labels)
            coverage = np.mean((labels >= predicts_l) & (labels <= predicts_u))
            interval_width = np.mean(np.abs(predicts_u - predicts_l))

            coverages.append(coverage)
            interval_widths.append(interval_width)
                
    return coverages, interval_widths



def mf_evaluate(metric, data_loader, test_model, device="cpu", params=None, alpha=0.1, standardize=False):
    test_model.eval()
    with torch.no_grad():
        if metric == "mse":
            labels, predicts = list(), list()
            for index, (uid, iid, rating) in enumerate(data_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict = test_model.predict(uid, iid)
                predict = params["min_val"] + predict * (params["max_val"] - params["min_val"])
                labels.extend(rating.tolist())
                predicts.extend(predict.tolist())
            mse = mean_squared_error(predicts, labels)
            return mse

        elif metric == "mpe":
            labels, predicts = list(), list()
            for index, (uid, iid, rating) in enumerate(data_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict = test_model.predict(uid, iid)
                if standardize:
                    predict = params["min_val"] + predict * (params["max_val"] - params["min_val"]) # reverse standardize?
                labels.extend(rating.tolist())
                predicts.extend(predict.tolist())
            mpe = mean_pinball_loss(predicts, labels, alpha=alpha)
            return mpe

        elif metric == "ndcg":
            uids, iids, predicts, labels = list(), list(), list(), list()
            for index, (uid, iid, rating) in enumerate(data_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict = test_model.predict(uid, iid)
                uids.extend(uid.cpu())
                iids.extend(iid.cpu())
                predicts.extend(predict.cpu())
                labels.extend(rating.cpu())
            label_matrix = csr_matrix((np.array(labels), (np.array(uids), np.array(iids))))
            label_matrix.eliminate_zeros()
            valid_rows = np.unique(label_matrix.nonzero()[0])
            label_matrix = label_matrix[valid_rows].toarray()

            predict_matrix = csr_matrix((np.array(predicts), (np.array(uids), np.array(iids))))
            predict_matrix = predict_matrix[valid_rows]
            predict_matrix.data += 1 << 10
            predict_matrix = predict_matrix.toarray()

            # ndcg = NDCG_binary_at_k_batch(predict_matrix1, label_matrix1, k=params["k"]).mean()
            # recall = Recall_at_k_batch(predict_matrix1, label_matrix1, k=params["k"]).mean()
            if device == "cpu":
                ndcg = NDCG_binary_at_k_batch(predict_matrix, label_matrix, k=params["k"]).mean()
                recall = Recall_at_k_batch(predict_matrix, label_matrix, k=params["k"]).mean()
                return ndcg, recall
            else:
                res = cal_ndcg_recall_parallel(2, predict_matrix, label_matrix, params["k"]).mean(axis=0)
                return res[0], res[1]


class MFRatingDataset(Dataset):
    def __init__(self, uid, iid, rating, require_index=False):
        self.uid = uid
        self.iid = iid
        self.rating = rating
        self.index = None
        if require_index:
            self.index = np.arange(0, self.uid.shape[0])

    def __getitem__(self, index):
        if self.index is None:
            return self.uid[index], self.iid[index], self.rating[index]
        else:
            return self.uid[index], self.iid[index], self.rating[index], self.index[index]

    def __len__(self):
        return len(self.rating)

def construct_wcp_mf_dataloader(config, device, require_index=False):
    # ratios are determined by train_ratio and test_ratio

    data_params = config["data_params"]
    standardize = config["standardize"]

    # naive method only use randomized data for tr/cal/ts
    # train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
    #                                                         data_params["random_path"],
    #                                                         test_ratio=data_params["test_ratio"])

    train_obs_mat, val_obs_mat, test_obs_mat = construct_rating_dataset_for_naive(
                                                            data_params["train_path"],
                                                            # data_params["random_path"],
                                                            train_ratio=0.5,
                                                            test_ratio=0.01)
    
    train_int_mat, val_int_mat, test_int_mat = construct_rating_dataset_for_naive(
                                                            # data_params["train_path"],
                                                            data_params["random_path"],
                                                            train_ratio=data_params["train_ratio"],
                                                            test_ratio=data_params["test_ratio"])
    
    n_train_obs, n_val_obs, n_test_obs = len(train_obs_mat), len(val_obs_mat), len(test_obs_mat)
    n_train_int, n_val_int, n_test_int = len(train_int_mat), len(val_int_mat), len(test_int_mat)

    print(f"Obs Sample Size: n_train_obs: {n_train_obs}, n_val_obs: {n_val_obs}, n_test_obs: {n_test_obs}")
    print(f"Int Sample Size: n_train_int: {n_train_int}, n_val_int: {n_val_int}, n_test_obs: {n_test_int}")

    n_users = train_obs_mat[:, 0].astype(int).max() + 1
    n_items = train_obs_mat[:, 1].astype(int).max() + 1

    min_val, max_val = data_params["min_val"], data_params["max_val"]
    threshold = data_params["threshold"]

    if standardize:
        if config["metric"] in ["mse","mpe"]:
            train_obs_ratings = ((train_obs_mat[:, 2] - min_val) / (max_val - min_val)).astype(np.float32)
            train_int_ratings = ((train_int_mat[:, 2] - min_val) / (max_val - min_val)).astype(np.float32)
            evaluation_params = {
                "min_val": min_val,
                "max_val": max_val,
                "n_items": n_items
            }
        else:
            train_obs_ratings = (train_obs_mat[:, 2] >= threshold).astype(np.float32)
            train_int_ratings = (train_int_mat[:, 2] >= threshold).astype(np.float32)
            val_obs_mat[:, 2] = val_obs_mat[:, 2] >= threshold
            # train_int_mat[:,2] = train_int_mat[:,2]>=threshold
            test_int_mat[:, 2] = test_int_mat[:, 2] >= threshold

            evaluation_params = {
                "k": config["topk"]
            }
    else:
        train_obs_ratings = train_obs_mat[:, 2]
        train_int_ratings = train_int_mat[:, 2]

        evaluation_params = {
                # "min_val": min_val,
                # "max_val": max_val,
                "n_items": n_items
            }

    # train_obs_loader, val_obs_loader, train_int_loader, test_int_loader = get_dataloader_wcp(train_obs_mat,
    #                                                        train_obs_ratings,
    #                                                        val_obs_mat,
    #                                                        train_int_mat,
    #                                                        test_int_mat,
    #                                                        config["batch_size"],
    #                                                        require_index=require_index)
        
    train_obs_loader, val_obs_loader, _ = get_dataloader(train_obs_mat,
                                                           train_obs_ratings,
                                                           val_obs_mat,
                                                           test_obs_mat,
                                                           config["batch_size"],
                                                           require_index=require_index)
    
    train_int_loader, val_int_loader, test_int_loader = get_dataloader(train_int_mat,
                                                           train_int_ratings,
                                                           val_int_mat,
                                                           test_int_mat,
                                                           config["batch_size"],
                                                           require_index=require_index)
    
    return train_obs_loader, val_obs_loader, train_int_loader, val_int_loader, test_int_loader, evaluation_params, n_users, n_items


def construct_naive_mf_dataloader(config, device, require_index=False):
    # ratios are determined by train_ratio and test_ratio

    standardize = config["standardize"]
    data_params = config["data_params"]

    # naive method only use randomized data for tr/cal/ts
    # train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
    #                                                         data_params["random_path"],
    #                                                         test_ratio=data_params["test_ratio"])

    train_mat, val_mat, test_mat = construct_rating_dataset_for_naive(
                                                            data_params["random_path"],
                                                            train_ratio=data_params["train_ratio"],
                                                            test_ratio=data_params["test_ratio"])

    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    min_val, max_val = data_params["min_val"], data_params["max_val"]
    threshold = data_params["threshold"]

    if standardize:
        if config["metric"] in ["mse", "mpe"]:
            train_ratings = ((train_mat[:, 2] - min_val) / (max_val - min_val)).astype(np.float32)
            evaluation_params = {
                "min_val": min_val,
                "max_val": max_val,
                "n_items": n_items
            }
        else:
            train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
            val_mat[:, 2] = val_mat[:, 2] >= threshold
            test_mat[:, 2] = test_mat[:, 2] >= threshold

            evaluation_params = {
                "k": config["topk"]
            }
    else:
        train_ratings = train_mat[:, 2]
        
    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"],
                                                           require_index=require_index)
    
    return train_loader, val_loader, test_loader, evaluation_params, n_users, n_items

def construct_mf_dataloader(config, device, require_index=False):
    data_params = config["data_params"]

    train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])
    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    min_val, max_val = data_params["min_val"], data_params["max_val"]
    threshold = data_params["threshold"]

    if config["metric"] == "mse":
        train_ratings = ((train_mat[:, 2] - min_val) / (max_val - min_val)).astype(np.float32)
        evaluation_params = {
            "min_val": min_val,
            "max_val": max_val,
            "n_items": n_items
        }
    else:
        train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
        val_mat[:, 2] = val_mat[:, 2] >= threshold
        test_mat[:, 2] = test_mat[:, 2] >= threshold

        evaluation_params = {
            "k": config["topk"]
        }

    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"],
                                                           require_index=require_index)
    return train_loader, val_loader, test_loader, evaluation_params, n_users, n_items


def get_dataloader(train_mat, train_ratings, val_mat, test_mat, batch_size, require_index=False, num_workers=5,
                   pin_memory=True):
    train_dataset = MFRatingDataset(train_mat[:, 0].astype(int),
                                    train_mat[:, 1].astype(int),
                                    train_ratings,
                                    require_index)
    val_dataset = MFRatingDataset(val_mat[:, 0].astype(int),
                                  val_mat[:, 1].astype(int),
                                  val_mat[:, 2])
    test_dataset = MFRatingDataset(test_mat[:, 0].astype(int),
                                   test_mat[:, 1].astype(int),
                                   test_mat[:, 2])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

def load_uniform_data_from_np(ratio, array, shape):
    size = int(ratio * array.shape[0])
    index = np.random.permutation(np.arange(array.shape[0])[:size])
    rows, cols, rating = array[index, 0], array[index, 1], array[index, 2]
    return csr_matrix(
        (rating, (rows, cols)), shape=shape
    ), index


def construct_ips_dataloader(config, device):
    data_params = config["data_params"]
    train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])
    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    min_val, max_val = data_params["min_val"], data_params["max_val"]
    threshold = data_params["threshold"]

    if config["metric"] == "mse":
        train_ratings = ((train_mat[:, 2] - min_val) / (max_val - min_val)).astype(np.float32)
        evaluation_params = {
            "min_val": min_val,
            "max_val": max_val,
            "n_items": n_items
        }
    else:
        train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
        val_mat[:, 2] = val_mat[:, 2] >= threshold
        test_mat[:, 2] = test_mat[:, 2] >= threshold

        evaluation_params = {
            "k": config["topk"]
        }
    uniform_data, index = load_uniform_data_from_np(0.166, val_mat, shape=(n_users, n_items))
    val_mat = np.delete(val_mat, index, axis=0)

    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"])

    def Naive_Bayes_Propensity(train, unif):
        # follow [1] Jiawei Chen et, al, AutoDebias: Learning to Debias for Recommendation 2021SIGIR and
        # [2] Tobias Schnabel, et, al, Recommendations as Treatments: Debiasing Learning and Evaluation
        P_Oeq1 = train.getnnz() / (train.shape[0] * train.shape[1])
        train.data[train.data < threshold] = 0
        train.data[train.data >= threshold] = 1
        # unif.data[unif.data < threshold] = 0
        # unif.data[unif.data > threshold] = 1

        y_unique = np.unique(train.data)
        P_y_givenO = np.zeros(y_unique.shape)
        P_y = np.zeros(y_unique.shape)

        for i in range(len(y_unique)):
            P_y_givenO[i] = np.sum(train.data == y_unique[i]) / np.sum(
                np.ones(train.data.shape))
            P_y[i] = np.sum(unif.data == y_unique[i]) / np.sum(np.ones(unif.data.shape))
        Propensity = P_y_givenO * P_Oeq1 / P_y
        Propensity = Propensity * (np.ones((n_items, 2)))

        return y_unique, Propensity

    y_unique, Propensity = Naive_Bayes_Propensity(np_to_csr(train_mat), uniform_data)
    InvP = torch.reciprocal(torch.tensor(Propensity, dtype=torch.float)).to(device)

    return train_loader, val_loader, test_loader, evaluation_params, n_users, n_items, y_unique, InvP


def read_best_params(model, key_name, sr=0.1, cr=2.0, tr=0.0):
    dir_prefix = os.getcwd()
    file_path = "/res/ndcg/sim_{}.json".format(key_name)
    if key_name == "sr":
        key = sr
    elif key_name == "cr":
        key = cr
    else:
        key = tr
    with open(dir_prefix + file_path, "r") as f:
        config = json.load(f)
        for model_config in config["models"]:
            if model == model_config["name"]:
                for param in model_config["params"]:
                    if param[key_name] == key:
                        return param
    raise Exception("invalid ")
