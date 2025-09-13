# Ranking model (deep learning, gradient boosting, etc.)
"""
ranker.py
---------
Implements a deep ranking model (two-tower architecture with user/item embeddings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ranker(nn.Module):
    """
    Two-tower ranking model: learns embeddings for users and items, combines with MLP.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        x = torch.cat([u, i], dim=1)
        return torch.sigmoid(self.mlp(x)).squeeze()


if __name__ == "__main__":
    model = Ranker(num_users=1000, num_items=5000)
    users = torch.randint(0, 1000, (16,))
    items = torch.randint(0, 5000, (16,))
    scores = model(users, items)
    print("Scores:", scores)
