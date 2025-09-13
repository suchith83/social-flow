# Recommendation / personalization model
# ============================
# File: model.py
# ============================
import torch
import torch.nn as nn

class TwoTowerRecommender(nn.Module):
    """Two-tower user–item embedding model for personalization."""

    def __init__(self, user_vocab_size, item_vocab_size, embedding_dim, hidden_units, dropout=0.3):
        super().__init__()
        self.user_emb = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_emb = nn.Embedding(item_vocab_size, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user, item, labels=None):
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=-1)
        logits = self.mlp(x).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
