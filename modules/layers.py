#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/12

import torch.nn as nn

Embedding = nn.Embedding


class ColorMLP(nn.Module):

    def __init__(self, in_dim:int=32, hidden_dim:int=32):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


class OccluMLP(nn.Module):

    def __init__(self, in_dim:int=32, hidden_dim:int=32):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.mlp(x)
