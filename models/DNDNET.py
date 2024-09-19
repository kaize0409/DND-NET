import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import torch
import torch.nn.functional as F

def feature_propagation(adj_h, features, K, alpha, args=None, only_k=True):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj_h = adj_h.to(device)
    features_prop = features.clone()
    for i in range(1, K + 1):
        features_prop = torch.sparse.mm(adj_h, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
    return features_prop

def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj

def process_adj(adj):
    adj_h = adj
    adj_h.setdiag(1)
    adj_h = adj_h + adj_h.T.multiply(adj_h.T > adj_h) - adj_h.multiply(adj_h.T > adj_h)
    adj_h = normalize_adj(adj_h)
    adj_h = sparse_mx_to_torch_sparse_tensor(adj_h)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, adj_h

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x

class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out

class DnDNet(nn.Module):
    """
    The masked PPR
    """
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout=0.5, use_bn=False):
        super(DnDNet, self).__init__()

        self.encoder = MLP_encoder(nfeat=nfeat,
                                 nhid=nhid,
                                 dropout=dropout)

        self.classifier = MLP_classifier(nfeat=nhid,
                                         nclass=nclass,
                                         dropout=dropout)

        # adaptive noise to be added.
        self.noise= nn.Parameter(torch.randn(nnodes, nfeat))

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    def forward(self, g, noisy_rate=0.1, n_id=None):
        
        x_pure, y_pure, z_pure = self.forward_pure(g)
        noisy_g = self.adding_noise(g, noisy_rate=noisy_rate, n_id=n_id)
        x_noisy, y_noisy, z_noisy  = self.forward_noisy(noisy_g)

        return x_pure, y_pure, z_pure, x_noisy, y_noisy, z_noisy
    
    def adding_noise(self, g, noisy_rate, n_id=None):

        noisy_g = g.clone()
        if n_id is None:
            noisy_g += torch.sign(noisy_g) * F.normalize(self.noise) * noisy_rate
        else:
            noisy_g += torch.sign(noisy_g) * F.normalize(self.noise[n_id]) * noisy_rate
        return noisy_g

    def forward_pure(self, x):
        if self.use_bn:
            x = self.bn1(x)
        h = self.encoder(x)
        if self.use_bn:
            h = self.bn2(h)
        y, z = self.classifier(h)

        return h, y, z

    def forward_noisy(self, x):
        if self.use_bn:
            x = self.bn1(x)
        h = self.encoder(x)
        if self.use_bn:
            h = self.bn2(h)
        y, z = self.classifier(h)
        return h, y, z