# Load datasets for content analysis
# ============================
# File: data_loader.py
# ============================
import torch
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoTokenizer
from typing import Dict, Any

class ContentDataset(Dataset):
    """Custom Dataset for content analysis."""

    def __init__(self, file_path: str, tokenizer: AutoTokenizer, text_col: str, label_col: str, max_len: int = 128):
        self.samples = [json.loads(line) for line in open(file_path, "r", encoding="utf-8")]
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        text = item[self.text_col]
        label = item[self.label_col]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

def create_dataloader(file_path: str, tokenizer, text_col: str, label_col: str, batch_size: int, num_workers: int, shuffle: bool):
    dataset = ContentDataset(file_path, tokenizer, text_col, label_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
