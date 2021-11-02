#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 supdev <ljyswzxhdtz@gmail.com>
#
# Distributed under terms of the MIT license.

"""
A implementation of ogbn datasets
"""
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import GATConv


root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]


train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx = train_idx, 
        sizes[10, 10, 10], batch_size=512, 
        shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], 
        batch_size=1024, shuffle=False, 
        num_workers=12)


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, heads=8):
        super(GAT, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_dim, hid_dim, heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(heads * hid_dim, hid_dim, heads)
                    )
        self.convs.append(heads * hid_dim, out_dim, hedas, concat=False)

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(dataset.num_features, hid_dim * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                    Lin(hid_dim * heads, hid_dim * heads)
                    )
        self.skips.append(Lin(hid_dim * heads, out_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_laoder:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())
                
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(dataset.num_features, 128, dataset.num_classes, num_layers=3, heads=4)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbat = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, u[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()
    
    loss = total_loss / len(train_loader)
    approx_acc = tital_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()


    out = model.inference(x)
    
    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']], 
        'y_pred': y_pred[split_idx['train']],
        })['acc']

    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']], 
        'y_pred': y_pred[split_idx['valid']], 
        })['acc']

    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']], 
        'y_pred': y_true[split_idx['test']], 
        })['acc']

    return train_acc, val_acc, test_acc


test_accs = []
for run in range(1, 11):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 101):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 50 and epoch % 10 == 0:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

    test_accs.append(final_test_acc)


test_acc = torch.tensor(test_accs)
print('=========================')
print(f'Final Test: {test_acc.mean():.4f} +- {test_acc.std():.4f}')
