"""
Deep Learning recommendation models.
Supports:
- Neural Collaborative Filtering (NCF)
- Deep Matrix Factorization
"""

import torch
import torch.nn as nn


class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF).
    Combines embeddings with MLP layers.
    """

    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128, 64, 32]):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(x).squeeze()


class DeepMatrixFactorization(nn.Module):
    """
    Deep Matrix Factorization (DMF).
    """

    def __init__(self, num_users, num_items, embedding_dim=64):
        super(DeepMatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        return (user_vec * item_vec).sum(dim=-1)
