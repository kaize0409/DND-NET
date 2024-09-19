import numpy as np
import torch
from models.DNDNET import DnDNet
from utils import *
from train import *
from best_paras import best_pars
from arguments import arg_parser
import csv

import warnings as warnings
warnings.filterwarnings("ignore")

def run_main_arxiv():
    args = arg_parser()

    if args.dataset in ['cora_ml', 'cora', 'citeseer', 'cs', 'pubmed']:
        args.__dict__.update(best_pars[f"{args.dataset}_best_{args.noise}_{int(args.ptb_rate*10):02d}"])
    args.dataset = 'arxiv'
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    data, meta = load_data(args.dataset)

    if args.dataset == 'arxiv':
        idx_train, idx_val, idx_test = data.split_idx['train'], data.split_idx['valid'], data.split_idx['test']
    degrees = pyg_utils.degree(data.edge_index[0])

    adj = edge_index_to_sparse_mx(data.edge_index, data.num_nodes)
    adj, adj_h = process_adj(adj)
    g = feature_propagation(adj_h, data.x, args.lg_k, args.alpha)
    
    accs = []
    for run in range(1, args.runs+1):
        setup_seed(run)
        data = data.to("cpu")
        features, labels = data.x, data.y.numpy()
        if args.dataset != 'arxiv':
            split = get_split(data.y, data.num_nodes, data.num_classes, args.label_rate, args.val_rate)
            idx_train, idx_val, idx_test = split

        train_labels = labels[idx_train]
        args.num_classes = meta['num_classes']
        noise_y, P = noisify_with_P(train_labels, meta['num_classes'], args.ptb_rate, 10, args.noise)
        noise_labels = labels.copy()
        noise_labels[idx_train] = noise_y
        labels = torch.LongTensor(np.array(noise_labels)).to(device)
        
        model = DnDNet(nnodes=data.x.shape[0], nfeat=data.x.shape[1], nhid=args.hidden, nclass=meta['num_classes'], dropout=args.dropout)
        model = model.to(device)
        data = data.to(device)

        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
        
        acc_val_all, acc_test_all = train_DnDNet_arxiv(model, optimizer, g, data, idx_train, idx_val, idx_test, labels, args)
        
        best_epoch = np.argmax(acc_val_all)
        print('[RUN{}] best epoch: {}, val acc: {:.4f}, test acc: {:.4f}'.format(run, (1+best_epoch * args.eval_freq),acc_val_all[best_epoch], acc_test_all[best_epoch]))

        accs.append(acc_test_all[best_epoch])

    print('[FINAL RESULT] test acc: {:.4f}+-{:.4f}'.format(np.mean(accs), np.std(accs)))

    res = [{
        'noise_ratio' : args.ptb_rate,
        'alpha' : args.alpha,
        'ce_b' : args.ce_b,
        'lambda_cr' : args.lambda_cr,
        'lambda_ncr' : args.lambda_ncr,
        'beta' : args.beta,
        'acc_mean' : '{:.4f}'.format(np.mean(accs)*100),
        'std': '{:.4f}'.format(np.std(accs)*100)
    }]

    with open(f'{args.dataset}.csv', 'a', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=res[0].keys())
        csv_writer.writerows(res)

if __name__ == '__main__':
    run_main_arxiv()