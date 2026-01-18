#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 3:26 下午
# @Author  : xuenan
# @Site    : 
# @File    : GCNEmbedderDecoder.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.mean(x, dim=0)


class GraphDecoder(nn.Module):
    def __init__(self, h_dim, num_nodes):
        super().__init__()
        self.node_proj = nn.Linear(h_dim, num_nodes * num_nodes)
        self.num_nodes = num_nodes

    def forward(self, h):
        B = h.size(0)
        A = self.node_proj(h).view(B, self.num_nodes, self.num_nodes)
        return torch.sigmoid((A + A.transpose(1, 2)) / 2)


class GCNEmbedderNew(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data, mask=None):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        if mask is not None:
            x = x[mask]  # 只平均有效节点
        return torch.mean(x, dim=0)


class GraphDecoderNew(nn.Module):
    def __init__(self, h_dim, num_nodes):
        super().__init__()
        self.node_proj = nn.Linear(h_dim, num_nodes * num_nodes)
        self.num_nodes = num_nodes

    def forward(self, h, mask=None):
        B = h.size(0)
        A = self.node_proj(h).view(B, self.num_nodes, self.num_nodes)
        A = torch.sigmoid((A + A.transpose(1, 2)) / 2)

        if mask is not None:
            # mask: [B, N], bool
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, N, N]
            A = A * mask.float()

        return A


class GCNEmbedderMulti(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return torch.mean(x, dim=0)


class GraphDecoderMulti(nn.Module):
    def __init__(self, h_dim, num_nodes):
        super().__init__()
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, num_nodes * num_nodes)
        self.num_nodes = num_nodes

    def forward(self, h):
        x = F.relu(self.fc1(h))
        A = self.fc2(x).view(-1, self.num_nodes, self.num_nodes)
        A = (A + A.transpose(1, 2)) / 2  # 保对称
        return torch.sigmoid(A)
