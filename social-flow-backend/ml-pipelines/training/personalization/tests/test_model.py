# Unit tests for model
# ============================
# File: tests/test_model.py
# ============================
import torch
from ml_pipelines.training.personalization.model import TwoTowerRecommender

def test_two_tower_forward():
    model = TwoTowerRecommender(user_vocab_size=10, item_vocab_size=20, embedding_dim=8, hidden_units=[16, 8])
    out = model(torch.tensor([1,2]), torch.tensor([3,4]), labels=torch.tensor([1.0, 0.0]))
    assert "logits" in out and "loss" in out
