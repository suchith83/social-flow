# Load datasets for personalization tasks
# ============================
# File: data_loader.py
# ============================
import torch
from torch.utils.data import Dataset, DataLoader
import json

class InteractionDataset(Dataset):
    """User–item interaction dataset."""

    def __init__(self, file_path, user_col, item_col, label_col):
        self.samples = [json.loads(l) for l in open(file_path, "r", encoding="utf-8")]
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user": torch.tensor(s[self.user_col], dtype=torch.long),
            "item": torch.tensor(s[self.item_col], dtype=torch.long),
            "label": torch.tensor(s[self.label_col], dtype=torch.float)
        }

def create_dataloader(file_path, user_col, item_col, label_col, batch_size, num_workers, shuffle):
    dataset = InteractionDataset(file_path, user_col, item_col, label_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
