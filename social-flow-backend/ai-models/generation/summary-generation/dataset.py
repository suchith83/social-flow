"""
Dataset utilities for summarization.
Supports two modes:
- extractive: labels are indices of sentences to pick
- abstractive: labels are summary text strings
"""

from torch.utils.data import Dataset
import torch
import pandas as pd
from typing import Optional, List
from .utils import logger

class AbstractiveSummaryDataset(Dataset):
    """
    Dataset for abstractive summarization.
    Expects a DataFrame with columns: 'document' and 'summary'
    Tokenization handled externally by a tokenizer object with .encode() method.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_input_len: int, max_target_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        document = str(row["document"])
        summary = str(row["summary"])

        input_ids = self.tokenizer.encode(document, max_length=self.max_input_len)
        target_ids = self.tokenizer.encode(summary, max_length=self.max_target_len)

        # add SOS/EOS if tokenizer supports them (assume numeric ids exist)
        sos = self.tokenizer.stoi.get(self.tokenizer.sos_token, None)
        eos = self.tokenizer.stoi.get(self.tokenizer.eos_token, None)
        if sos is not None:
            target_ids = [sos] + target_ids
        if eos is not None:
            target_ids = target_ids + [eos]

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        return input_tensor, target_tensor

class ExtractiveSummaryDataset(Dataset):
    """
    Dataset for extractive summarization.
    Expects DataFrame with columns: 'document_sentences' (list of sentences) and 'labels' (list of indices)
    Returns: sentence encodings and binary label vector per sentence
    """

    def __init__(self, df: pd.DataFrame, sentence_encoder, max_sentences: int = 50):
        self.df = df.reset_index(drop=True)
        self.sentence_encoder = sentence_encoder  # function: sentence -> vector tensor
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sents = list(row["document_sentences"])[: self.max_sentences]
        encoded = [self.sentence_encoder(s) for s in sents]  # list of tensors
        # pad sentences to max_sentences with zeros
        if len(encoded) < self.max_sentences:
            pad = [torch.zeros_like(encoded[0]) for _ in range(self.max_sentences - len(encoded))] if encoded else [torch.zeros(768) for _ in range(self.max_sentences)]
            encoded += pad
        enc_tensor = torch.stack(encoded)  # shape (max_sentences, feat_dim)

        labels = row.get("labels", [])
        label_vec = torch.zeros(self.max_sentences, dtype=torch.float32)
        for i in labels:
            if i < self.max_sentences:
                label_vec[i] = 1.0
        return enc_tensor, label_vec
