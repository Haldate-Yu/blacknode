#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 supdev <ljyswzxhdtz@gmail.com>
#
# Distributed under terms of the MIT license.

"""
A File to Load Dataset
"""
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI, Planetoid
from torch_geometric.loader import DataLoader


# for cora, citeseer, pubmed node classification
def load_citation(dataname, path):
    dataset = Planetoid(path, dataname)
    data = dataset[0]
    return dataset, data


# for cora, citeseer, pubmed link prediction
def load_citation_for_link_prediction(dataname, path):
    transform = T.Compose([
        T.NormalizeFeatures(), 
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, 
                        add_negative_train_samples=False), 
    ])
    dataset = Planetoid(path, name=dataname, transform=transform)
    train_data, val_data, test_data = dataset[0]

    return train_data, val_data, test_data


def load_ppi(path, batch_size=4):
    pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
    
    train_dataset = PPI(path, split='train', pre_transform=pre_transform)
    val_dataset = PPI(path, split='val', pre_transform=pre_transform)
    test_dataset = PPI(path, split='test', pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
