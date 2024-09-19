import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import torch_geometric.transforms as T
import torch.nn.functional as F
from scipy.sparse import csc_matrix, coo_matrix
import scipy.sparse as sp
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import LabelPropagation

import torch
import torch.nn.functional as F
import numpy as np
import random
import torch_geometric
from typing import Optional, Tuple
import time

EPS = 1e-5

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)

# feature propagation - from D2PT
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

def get_indices(edge_index, num_nodes):
    neighborhoods = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        neighborhoods[src].append(dst)

    return neighborhoods

def get_neighbors(edge_index, num_nodes):

    device = edge_index.device

    neighborhoods = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        neighborhoods[src].append(dst)

    max_neighbor = max(len(lst) for lst in neighborhoods)

    ind_nid_m = torch.zeros((num_nodes, max_neighbor)) # stored node id nxk
    ind_m = torch.zeros((num_nodes, num_nodes)) # stored boolean value at each position nxn

    nb_m = torch.zeros((num_nodes,1))

    for i in range(num_nodes):
        neighbors = torch.tensor(neighborhoods[i], dtype=torch.int).to(device)
        indices = sorted(neighbors.tolist())
        ind_nid_m[i,:len(indices)] = torch.tensor(indices)
        nb_m[i] = len(indices)
        
        ind_m[i,indices] = 1
    ind_nid_m = ind_nid_m.long()


    return ind_nid_m, ind_m, nb_m

def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:valid_num+test_num]

    return train_idx, valid_idx, test_idx

def get_split(y, n_nodes, n_classes, label_rate, val_rate):
    label_num_per_class = int(label_rate * n_nodes / n_classes + 1)
    val_num = int(val_rate * n_nodes + 1)
    test_num = n_nodes - label_num_per_class * n_classes - val_num

    idx_train, idx_val, idx_test = class_rand_splits(torch.LongTensor(y), label_num_per_class,
                                                                    valid_num=val_num, test_num=test_num)

    return idx_train, idx_val, idx_test


def load_data(dset, normalize_features=True):
    
    path = osp.join('.', 'Data', dset)

    if dset in ['pubmed']:
        dataset = Planetoid(path, dset)
        data = dataset[0]

    if dset in ['cs']:
        dataset = Coauthor(path, dset)
        data = dataset[0]

    if dset in ['arxiv']:
        dataset = PygNodePropPredDataset(root="./Data", name = 'ogbn-arxiv')
        data = dataset[0]
        data.split_idx = dataset.get_idx_split()
    
    if dset in ['citeseer']:
        data = torch.load(f"./Data/citeseer.pt")
        data.y = torch.squeeze(data.y)
        data.num_classes = torch.unique(data.y).shape[0]
        if normalize_features:
            data.transform = T.NormalizeFeatures()
        meta = {'num_classes': data.num_classes}
        return data, meta
    if dset in ['cora_ml']:
        data = torch.load(f"./Data/cora_ml.pt")
        data.y = torch.squeeze(data.y)
        data.num_classes = torch.unique(data.y).shape[0]

        if normalize_features:
            data.transform = T.NormalizeFeatures()
        meta = {'num_classes': data.num_classes}
        return data, meta
    else:
        assert Exception

    data.y = torch.squeeze(data.y)
    data.num_classes = dataset.num_classes
    if normalize_features:
        data.transform = T.NormalizeFeatures()

    meta = {'num_classes': dataset.num_classes}

    return data, meta

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_pair_p(size, noise):
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def noisify(y, p_minus, p_plus=None, random_state=0):
    """ Flip labels with probability p_minus.
    If p_plus is given too, the function flips with asymmetric probability.
    """

    assert np.all(np.abs(y) == 1)

    m = y.shape[0]
    new_y = y.copy()
    coin = np.random.RandomState(random_state)

    if p_plus is None:
        p_plus = p_minus

    for idx in np.arange(m):
        if y[idx] == -1:
            if coin.binomial(n=1, p=p_minus, size=1) == 1:
                new_y[idx] = -new_y[idx]
        else:
            if coin.binomial(n=1, p=p_plus, size=1) == 1:
                new_y[idx] = -new_y[idx]

    return new_y

def noisify_with_P(y_train, nb_classes, noise, random_state=None,  noise_type='uniform'):

    if noise > 0.0:
        if noise_type=='uniform':
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P


def pre_compute_local_sim_(emb, edge_index, args, max_neighbor=30):
    device = emb.device
    src_emb = emb[edge_index[0]]
    dst_emb = emb[edge_index[1]]
    cosine_similarities = F.cosine_similarity(src_emb, dst_emb)
    num_nodes = emb.size(0)
    neighborhoods = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        neighborhoods[src].append(dst)

    ind_nid_m = torch.zeros((emb.shape[0], max_neighbor)) 
    ind_m = torch.zeros((emb.shape[0],emb.shape[0])) 
    
    sim_score_m = torch.zeros((emb.shape[0], max_neighbor))

    nb_m = torch.zeros((emb.shape[0],1))
    for i in range(num_nodes):
        neighbors = torch.tensor(neighborhoods[i], dtype=torch.int).to(device)
        sim_scores = cosine_similarities[edge_index[0] == i]
        sorted_indices = torch.argsort(sim_scores, descending=True).cpu().tolist()

        if sim_scores.shape[0] > max_neighbor:
            ind_nid_m[i] = torch.tensor(neighbors[:max_neighbor].tolist())
            sim_score_m[i] = torch.tensor(sim_scores[:max_neighbor].tolist())
            nb_m[i] = max_neighbor
        else:
            ind_nid_m[i,:neighbors.shape[0]] = neighbors 
            sim_score_m[i,:neighbors.shape[0]] = sim_scores 
            nb_m[i] = sim_scores.shape[0]
        ind_m[i,neighbors[sorted_indices].tolist()] = 1
    ind_nid_m = ind_nid_m.long()

    if args.zero == "yes":
        sim_score_m = torch.relu(sim_score_m)
    
    return ind_nid_m, ind_m, sim_score_m, nb_m


def neighbor_align(y_pure: torch.Tensor,
                    ind_nid_m : torch.Tensor,
                    nb_m : torch.Tensor,
                    data,
                    args,
                    unlabels,
                    epsilon: float = 1e-16,
                    tem: float = 0.1):

    device = y_pure.device
    tem = args.ncr_t
    neighbor_logits = y_pure[ind_nid_m] # n x k x |lb|
    neighbor_logits = torch.exp(neighbor_logits)
    nb_m = nb_m.squeeze() # n
    mask = torch.arange(neighbor_logits.size(1)).expand(neighbor_logits.size(0), neighbor_logits.size(2), neighbor_logits.size(1)).permute(0,2,1).to(nb_m.device) >= nb_m.view(-1, 1, 1)
    neighbor_logits[mask]=0

    if args.ncr_loss == "kl":
        """
        measure ncr loss based on kl-d
        """
        nonzero_rows = torch.any(neighbor_logits != 0, dim=2)
        mean = (neighbor_logits * nonzero_rows.unsqueeze(2)).sum(dim=1) / (nonzero_rows.sum(dim=1, keepdim=True).float() + epsilon)
        sharp_mean = (torch.pow(mean, 1./tem) / torch.sum(torch.pow(mean, 1./tem) + epsilon, dim=1, keepdim=True)).detach()

        if args.useunlabel == "yes":
            kl_loss = F.kl_div(y_pure, sharp_mean, reduction='none')[unlabels].sum(1)
            filtered_kl_loss = kl_loss[mean[unlabels].max(1)[0] > args.ncr_conf]
            local_ncr = torch.mean(filtered_kl_loss)
        else:
            local_ncr = torch.mean((-sharp_mean * torch.log_softmax(y_pure, dim=1)).sum(1)[torch.softmax(mean, dim=-1).max(1)[0] > args.ncr_conf])
    
    elif args.ncr_loss == "weighted_kl":
        nonzero_rows = torch.any(neighbor_logits != 0, dim=2)
        sim_score_m_l = data.sim_score_m
        weights = torch.div(sim_score_m_l, torch.sum(sim_score_m_l, dim=1, keepdim=True) + epsilon)
        weighted_mean = ((neighbor_logits * nonzero_rows.unsqueeze(2)) * weights.unsqueeze(2).to(device)).sum(dim=1) / nonzero_rows.sum(dim=1, keepdim=True).float()
        sharp_mean = (torch.pow(weighted_mean, 1./tem) / (torch.sum(torch.pow(weighted_mean, 1./tem), dim=1, keepdim=True) + epsilon)).detach()
        if args.useunlabel == "yes":
            local_ncr = (-sharp_mean * y_pure)
            local_ncr_unlabeled = local_ncr[unlabels].sum(1)
            local_ncr_filtered = local_ncr_unlabeled[torch.softmax(weighted_mean[unlabels], dim=-1).max(1)[0] > args.ncr_conf]
            local_ncr = torch.mean(local_ncr_filtered)
        else:
            local_ncr = torch.mean((-sharp_mean * torch.log_softmax(y_pure, dim=1)).sum(1)[torch.softmax(weighted_mean, dim=-1).max(1)[0] > args.ncr_conf] + epsilon) + epsilon
    else:
        raise ValueError(f"Unknown loss type: {args.ncr_loss}")

    return local_ncr

def neighbor_align_batch(adj, x, y_pure: torch.Tensor,
                    args,
                    batch_ul_mask,
                    epsilon: float = 1e-16,
                    tem: float = 0.1):

    device = y_pure.device
    tem = args.ncr_t
    p = torch.exp(y_pure)
    edge_list, e_id, adj_size = adj
    coo_matrix = sp.coo_matrix((torch.ones(edge_list.size(1)).cpu().numpy(), (edge_list[1].cpu().numpy(), edge_list[0].cpu().numpy())),shape=(adj_size[1], adj_size[0]))


    adj_matrix = torch.sparse_coo_tensor(
        torch.LongTensor([coo_matrix.row, coo_matrix.col]),
        torch.FloatTensor(coo_matrix.data),
        torch.Size(coo_matrix.shape))
    adj_matrix = adj_matrix.to(args.device)

    if args.ncr_loss == 'kl':
        mean = torch.sparse.mm(adj_matrix, y_pure)
        mean = mean / (adj_matrix.sum(dim=1).to_dense().view(-1,1) + epsilon)
        sharp_mean = (torch.pow(mean, 1./tem) / torch.sum(torch.pow(mean, 1./tem) + epsilon, dim=1, keepdim=True)).detach()
        if args.useunlabel == "yes":
            kl_loss = F.kl_div(y_pure, sharp_mean, reduction='none')[batch_ul_mask].sum(1)
            filtered_kl_loss = kl_loss[mean[batch_ul_mask].max(1)[0] > args.ncr_conf]
            local_ncr = torch.mean(filtered_kl_loss)
        else:
            local_ncr = torch.mean((-sharp_mean * torch.log_softmax(y_pure, dim=1)).sum(1)[torch.softmax(mean, dim=-1).max(1)[0] > args.ncr_conf])
        
    if args.ncr_loss == 'weighted_kl':
        dst_emb = x[:adj_size[1]]
        src_emb = x
        cosine_sim = torch.mm(F.normalize(dst_emb, dim=1), F.normalize(src_emb, dim=1).T)
        sim_matrix = torch.mul(adj_matrix, cosine_sim)

        weights = sim_matrix.to_dense() / (sim_matrix.sum(dim=1).to_dense().view(-1,1) + epsilon)

        weighted_mean = torch.mm(weights, p)
        denominator = torch.sum(torch.pow(weighted_mean, 1./tem), dim=1, keepdim=True) + epsilon
        sharp_mean = (torch.pow(weighted_mean, 1./tem) / (torch.sum(torch.pow(weighted_mean, 1./tem), dim=1, keepdim=True) + epsilon)).detach()
        if args.useunlabel == "yes":
            local_ncr = (-sharp_mean * y_pure[:adj_size[1]])
            local_ncr_unlabeled = local_ncr[batch_ul_mask].sum(1)
            local_ncr_filtered = local_ncr_unlabeled[torch.softmax(weighted_mean[batch_ul_mask], dim=-1).max(1)[0] > args.ncr_conf]
            local_ncr = torch.mean(local_ncr_filtered)
        else:
            local_ncr = torch.mean((-sharp_mean * torch.log_softmax(y_pure, dim=1)).sum(1)[torch.softmax(weighted_mean, dim=-1).max(1)[0] > args.ncr_conf] + epsilon) + epsilon
    else:
        raise ValueError(f"Unknown loss type: {args.ncr_loss}")

    return local_ncr

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def get_uncertainty(data, y_pure, args, epsilon=1e-16):

    device = y_pure.device
    p = torch.exp(y_pure)
    neighbor_logits = p[data.ind_nid_m] # n x k x |lb|
    neighbor_logits = torch.exp(neighbor_logits)
    nb_m = data.nb_m.squeeze() # n
    mask = torch.arange(neighbor_logits.size(1)).expand(neighbor_logits.size(0), neighbor_logits.size(2), neighbor_logits.size(1)).permute(0,2,1).to(nb_m.device) >= nb_m.view(-1, 1, 1)
    neighbor_logits[mask]=0

    # Create a mask for rows that are not all zeros
    nonzero_rows = torch.any(neighbor_logits != 0, dim=2)
    ptc = neighbor_logits * nonzero_rows.unsqueeze(2)
    ptc = ptc.mean(dim=1)
    hpt = entropy(ptc)
    w = torch.exp(-hpt/torch.log2(torch.tensor(data.num_classes)))
    return w

def get_uncertainty_batch(adj, y_pure, args, epsilon=1e-16):

    edge_list, e_id, adj_size = adj
    p = torch.exp(y_pure)
    coo_matrix = sp.coo_matrix((torch.ones(edge_list.size(1)).cpu().numpy(), (edge_list[1].cpu().numpy(), edge_list[0].cpu().numpy())),shape=(adj_size[1], adj_size[0]))

    adj_matrix = torch.sparse_coo_tensor(
        torch.LongTensor([coo_matrix.row, coo_matrix.col]),
        torch.FloatTensor(coo_matrix.data),
        torch.Size(coo_matrix.shape))
    adj_matrix = adj_matrix.to(args.device)

    ptc = torch.sparse.mm(adj_matrix, p)
    ptc = ptc / (adj_matrix.sum(dim=1).to_dense().view(-1,1) + epsilon)
    hpt = entropy(ptc)
    w = torch.exp(-hpt/torch.log2(torch.tensor(args.num_classes)))
    return w


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss
    
def fix_cr(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True, w=None):
    assert name in ['ce', 'l2']

    pseudo_label = torch.exp(logits_w)
    logits_s = torch.exp(logits_s)

    pseudo_label = logits_w.detach()
    if name == 'l2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    elif name == 'ce':
        # pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        if w is None:
            return masked_loss.mean() #, mask.mean()
        else:
            return (w * masked_loss).mean()
    else:
        assert Exception('Not Implemented consistency_loss')
