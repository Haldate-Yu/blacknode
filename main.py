#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 supdev <ljyswzxhdtz@gmail.com>
#
# Distributed under terms of the MIT license.

"""
A main file to train and test
"""
import torch
import torch.nn as nn
from torch_geometric.loader import RandomNodeSampler
import copy

from models import *
from dataset import *
from utils import *


dataname = 'cora'
path = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset, data = load_citation(dataname, path)
test_data = copy.deepcopy(data)

# subgraph method
test_part = RandomNodeSampler(test_data, num_parts=2, shuffle=False, num_workers=5)
subgraph = []
for item in test_part:
    subgraph.append(item)
test_data = subgraph[0]

print(test_data)

model = GCN(dataset=dataset, hidden=32, num_layers=2)

model.to(device)
data.to(device)
test_data.to(device)

criterion = nn.NLLLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, pred = torch.max(out[data.train_mask], dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    acc = correct / data.train_mask.sum().item()

    return loss.item(), acc


@torch.no_grad()
def test():
    model.eval()

    out = model(test_data)
    loss= criterion(out[test_data.test_mask], test_data.y[test_data.test_mask])
    
    _, pred = torch.max(out[test_data.test_mask], dim=1)
    correct = (pred == test_data.y[test_data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()

    return loss.item(), acc


if __name__ == '__main__':
    for epoch in range(100):
        loss, acc = train()
        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(epoch, loss, acc))


    test_loss, test_acc = test()
    print("test_loss: {:.4f} test_acc: {:.4f}".format(test_loss, test_acc))
