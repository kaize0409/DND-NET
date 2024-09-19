import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
import torch_geometric.utils as pyg_utils
import time
from early_stop import EarlyStopping, Stop_args
from torch_geometric.loader import NeighborSampler

def train_DndNet(model, optimizer, 
              g, data, 
              idx_train, idx_val, idx_test, 
              labels, args):
    acc_val_all = []
    acc_test_all = []
    unsup_idx = torch.cat((idx_val, idx_test))
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    
    with tqdm(total=args.epochs, desc='(Training main model)', disable = not args.verbose) as pbar:
        for epoch in range(1, 1+args.epochs):
            model.train()
            loss = 0
            h_pure, y_pure, z_pure, h_noisy, y_noisy, z_noisy = model(g, args.noisy_rate)

            with torch.no_grad():
                w = get_uncertainty(data, y_pure, args)

            loss += (1- args.ce_b) * (F.nll_loss(y_pure[idx_train], labels[idx_train]))
            loss += args.ce_b * (F.nll_loss(y_noisy[idx_train], labels[idx_train]))

            loss_cr = fix_cr(y_pure[unsup_idx], y_noisy[unsup_idx], name='ce', w=w[unsup_idx])
            loss += args.lambda_cr * loss_cr

            loss_ncr_pure = neighbor_align(y_pure, data.ind_nid_m, data.nb_m, data, args, unsup_idx)
            loss += args.lambda_ncr * loss_ncr_pure

            optimizer.zero_grad()
            loss.backward()
            end_time = time.time()


            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if epoch % args.eval_freq == 0:
                with torch.no_grad():
                    model.eval()
                    h_pure, y_pure, z_pure, h_noisy, y_noisy, z_noisy = model(g, args.noisy_rate)
                    y_final = args.beta * y_pure + (1-args.beta) * y_noisy

                    acc_val = accuracy(y_final[idx_val], labels[idx_val]).item()
                    acc_test = accuracy(y_final[idx_test], labels[idx_test]).item()

                    acc_val_all.append(acc_val)
                    acc_test_all.append(acc_test)
                    if args.verbose == True:
                        print('Epoch {}, val acc: {:.4f}, test acc: {:.4f}'.format(epoch, acc_val_all[-1], acc_test_all[-1]))

            pbar.set_postfix({'loss': loss.detach().cpu().item()})
            pbar.update()
            if early_stopping.check([acc_val], epoch):
                break
            
    return  acc_val_all, acc_test_all

def train_DnDNet_arxiv(model, optimizer, 
              g, data, 
              idx_train, idx_val, idx_test, 
              labels, args):

    acc_val_all = []
    acc_test_all = []
    unsup_idx = torch.cat((idx_val, idx_test))
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    
    train_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=args.batch_size, shuffle=False, num_workers=8)

    with tqdm(total=args.epochs, desc='(Training main model)', leave=False, dynamic_ncols=True, disable = not args.verbose) as pbar:
        for epoch in range(1, 1+args.epochs):
            model.train()
            for batch_size, n_id, adj in train_loader:
                loss = 0
                adj = adj.to(args.device)
                batch_lb = labels[n_id].to(args.device)
                x = data.x[n_id].to(args.device)
                optimizer.zero_grad()
                h_pure, y_pure, z_pure, h_noisy, y_noisy, z_noisy = model(x, args.noisy_rate, n_id=n_id)
                with torch.no_grad():
                    w = get_uncertainty_batch(adj, y_pure, args)

                batch_train_mask = torch.isin(n_id[:adj[2][1]], idx_train)
                batch_ul_mask = torch.logical_not(batch_train_mask)

                loss += (1- args.ce_b) * (F.nll_loss(y_pure[:adj[2][1]][batch_train_mask], batch_lb[:adj[2][1]][batch_train_mask]))
                loss += args.ce_b * (F.nll_loss(y_noisy[:adj[2][1]][batch_train_mask], batch_lb[:adj[2][1]][batch_train_mask]))

                loss_cr = fix_cr(y_pure[:adj[2][1]][batch_ul_mask], y_noisy[:adj[2][1]][batch_ul_mask], name='ce', w=w[batch_ul_mask])
                loss += args.lambda_cr * loss_cr

                loss_ncr_pure = neighbor_align_batch(adj, x, y_pure, args, batch_ul_mask)
                loss += args.lambda_ncr * loss_ncr_pure

                optimizer.zero_grad()
                loss.backward()
                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if epoch % args.eval_freq == 0:
                with torch.no_grad():
                    model.eval()
                    h_pure, y_pure, z_pure, h_noisy, y_noisy, z_noisy = model(g, args.noisy_rate)
                    y_final = args.beta * y_pure + (1-args.beta) * y_noisy

                    acc_val = accuracy(y_final[idx_val], labels[idx_val]).item()
                    acc_test = accuracy(y_final[idx_test], labels[idx_test]).item()

                    acc_val_all.append(acc_val)
                    acc_test_all.append(acc_test)
                    if args.verbose == True:
                        print('Epoch {}, val acc: {:.4f}, test acc: {:.4f}'.format(epoch, acc_val_all[-1], acc_test_all[-1]))

            pbar.set_postfix({'loss': loss.detach().cpu().item()})
            pbar.update()
            if early_stopping.check([acc_val], epoch):
                break
            
    return  acc_val_all, acc_test_all