#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 9:13 下午
# @Author  : xuenan
# @Site    : 
# @File    : GRU.py
# @Software: PyCharm
import torch
import torch.nn as nn


class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入格式为 (batch, seq, feature)
        )

        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        # GRU处理
        out, hidden = self.gru(x, hidden)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]

        # Dropout和全连接层
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


