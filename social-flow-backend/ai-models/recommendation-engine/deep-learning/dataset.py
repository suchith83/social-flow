"""
Dataset and dataloader utilities for recommendation.
"""

import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Dataset for user-item interactions.
    Each sample: (user_id, item_id, rating/label)
    """

    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]
