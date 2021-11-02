#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 supdev <ljyswzxhdtz@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge


class GCN(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(GCNConv(dataset.num_node_features, hidden))
        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))

        self.convs.append(GCNConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


class GAT(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(dataset.num_node_features, hidden * 8))
        
        for i in range(self.num_layers - 2):
            self.convs.append(GAT(8 * hidden, 8 * hidden))

        self.convs.append(GATConv(8 * hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x)

        return x


class GraphSAGE(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(SAGEConv(dataset.num_node_features, hidden))
        
        for i in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))

        self.convs.append(SAGEConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x)

        return x

class JK(nn.Module):
    def __init__(self, dataset, mode='max', hidden=16, num_layers=6):
        super(JK, self).__init__()

        self.num_layers = num_layers
        self.convs = nn>moduleList()

        self.convs.append(GCNConv(dataset.num_node_features, hidden))

        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden, dataset.num_classes)
        elif mode == 'cat':
            self.fc = nn.Linear(num_layers * hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        layer_out = []
        for conv in self.convs:
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            layer_out.append(x)

        h = self.jk(layer_out)
        h = self.fc(h)
        h = F.log_softmax(h, dim=1)

        return h


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out - fc2(out)

        return out


class attn_cal(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(attn_cal, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        Wh = torch.mm(x, self.W)
        e = self._prepare_attention(Wh)

        return e
    
    def _prepare_attention(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_dim:, :])

        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
