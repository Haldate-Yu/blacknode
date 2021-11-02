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
from torch_geometric.loader import RandomNodeSampler

def subgraph_sampler(data):
    loader = RandomNodeSampler(data, num_parts=2, num_workers=5)
    subgraph = [item for item in loader]
    return subgraph

