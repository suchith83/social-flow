"""
Deep Learning based Recommender System wrapper.
"""

import torch
import numpy as np
import pandas as pd
from .utils import get_device
from .config import TOP_K


class DeepLearningRecommender:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)

    def recommend(self, user_id: int, all_item_ids: list, known_items=set(), top_k=TOP_K):
        """
        Recommend items for a given user.
        :param user_id: target user
        :param all_item_ids: list of all item IDs
        :param known_items: items already interacted with
        """
        self.model.eval()
        user_tensor = torch.tensor([user_id] * len(all_item_ids), dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(all_item_ids, dtype=torch.long, device=self.device)

        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()

        item_scores = [(item, score) for item, score in zip(all_item_ids, scores) if item not in known_items]
        item_scores.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(item_scores[:top_k], columns=["item_id", "score"])
