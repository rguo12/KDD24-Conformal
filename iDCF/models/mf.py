import torch.nn as nn
import torch
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MF(nn.Module):
    def __init__(self, num_users, num_items, y_unique, InverP, embedding_size=100, dropout=0.):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)

        self.corY = y_unique

        # self.drop = nn.Dropout(dropout)

        self.invP = InverP

        self.device = DEVICE

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return (U * I).sum(1) + b_u + b_i + self.mean

    def predict(self, uid, iid):
        return self.forward(uid, iid)

    def get_embedding(self,u_id,i_id):
        U = self.user_emb(u_id)
        I = self.item_emb(i_id)
        return U, I
    
    def compute_ips_weights(self, u_id, i_id, y_train):

        weight = torch.ones_like(y_train).to(self.device)
        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP[i_id][y_train == self.corY[i], i]
        
        return weight






class MFwithFeature(nn.Module):
    def __init__(self, num_users, num_items, feature_size, embedding_size=100, dropout=0., device="cpu"):
        super(MFwithFeature, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
        
        self.feature_embs_u = []
        self.feature_embs_i = []
        for feature_dim in feature_size:
            emb_u = nn.Embedding(feature_dim, 32)
            emb_u.weight.data.uniform_(-0.01, 0.01)
            emb_i = nn.Embedding(num_items, 32)
            emb_i.weight.data.uniform_(-0.01, 0.01)
            self.feature_embs_i.append(emb_i.to(device))
            self.feature_embs_u.append(emb_u.to(device))


        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.drop = nn.Dropout(dropout)
        self.device = device

    def forward(self, u_id, i_id, features):
        U = self.drop(self.user_emb(u_id))
        b_u = self.user_bias(u_id).squeeze()
        I = self.drop(self.item_emb(i_id))
        b_i = self.item_bias(i_id).squeeze()
        Y = torch.zeros_like(u_id).to(self.device)
        for i in range(features.shape[1]):
            Y = Y + (self.feature_embs_u[i](features[:, i]) * self.feature_embs_i[i](i_id)).sum(1)
        return ((U) * I).sum(1) + b_u + b_i + self.mean + Y

