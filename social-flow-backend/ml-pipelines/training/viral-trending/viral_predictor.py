# Viral prediction model (deep + temporal features)
"""
viral_predictor.py
------------------
Deep neural network for viral prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViralPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze()
