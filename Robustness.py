#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 9:17 下午
# @Site    : 
# @File    : Robustness.py
# @Software: PyCharm
import torch.nn as nn


# === 可导鲁棒性预测器模块 ===
class RobustnessPredictor(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_nodes * num_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, A):  # 输入 [B, N, N]
        return self.fc(A).squeeze(-1)  # 输出 [B]
