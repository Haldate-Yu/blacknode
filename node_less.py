#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 supdev <ljyswzxhdtz@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Situation of missing nodes during test
"""
import torch
import torch.nn as nn
from torch_geometric.loader import RandomNodeSampler
import copy


from models import *
from dataset import *
from utils import *


dataname = 'pubmed'
path = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset, data = load_citation(dataname, path)


# subgraph method
subgraph = subgraph_sampler(data)
# print(test_data)
test_data = subgraph[0]
train_data = subgraph[1]

# print(train_data)
# print(set(train_data.edge_index[0]))
# print(set(train_data.edge_index[1]))
# union = set(set(train_data.edge_index[0]) | set(train_data.edge_index[1]))
# print(len(union))
# print(len(set(train_data.edge_index[0])))

model = GCN(dataset=dataset, hidden=32, num_layers=2)

model.to(device)
data.to(device)
test_data.to(device)
train_data.to(device)

criterion = nn.NLLLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()

    out = model(train_data)
    loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, pred = torch.max(out[train_data.train_mask], dim=1)
    correct = (pred == train_data.y[train_data.train_mask]).sum().item()
    acc = correct / train_data.train_mask.sum().item()

    return loss.item(), acc


@torch.no_grad()
def test():
    model.eval()

    out = model(test_data)
    loss = criterion(out[test_data.test_mask], test_data.y[test_data.test_mask])
    val_loss = criterion(out[test_data.val_mask], test_data.y[test_data.val_mask])
    
    _, pred = torch.max(out[test_data.test_mask], dim=1)
    correct = (pred == test_data.y[test_data.test_mask]).sum().item()
    acc = correct / test_data.test_mask.sum().item()
    
    _, val_pred = torch.max(out[test_data.val_mask], dim=1)
    val_correct = (val_pred == test_data.y[test_data.val_mask]).sum().item()
    val_acc = val_correct / test_data.test_mask.sum().item()

    return loss.item(), val_loss.item(), acc, val_acc


if __name__ == '__main__':
    test_accs = []

    for run in range(10):
        print('')
        print(f'Run {run:02d}')
        print('')

        model.reset_parameters()
        best_val_acc = final_test_acc = 0

        for epoch in range(100):
            loss, acc = train()
            print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(epoch, loss, acc))

            if epoch > 50 and epoch % 10 == 0:
                test_loss, val_loss, test_acc, val_acc = test()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
                print("test_loss: {:.4f} test_acc: {:.4f}".format(test_loss, test_acc))
        print('-----------------------')
        print(f'final test this run: {final_test_acc:.4f}')
        print('-----------------------')
        test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print('=========================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
