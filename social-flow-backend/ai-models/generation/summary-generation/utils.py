"""
Utilities: logging, seed control, IO helpers, tokenizer wrapper
"""

import logging
import random
import os
import pickle
from typing import List
import numpy as np
import torch
from .config import MODEL_DIR, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("summary_generation")

def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_pickle(obj, name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved: {path}")
    return path

def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded: {path}")
    return obj

class SimpleTokenizer:
    """
    Minimal whitespace + basic punctuation tokenizer with integer mapping.
    Use only if HuggingFace tokenizers are not available.
    For production use a pretrained tokenizer (BPE / SentencePiece).
    """

    def __init__(self, stoi=None, itos=None, unk_token="<UNK>", pad_token="<PAD>",
                 sos_token="<SOS>", eos_token="<EOS>"):
        self.stoi = stoi or {}
        self.itos = itos or {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        # ensure special tokens present
        for t in [pad_token, unk_token, sos_token, eos_token]:
            if t not in self.stoi:
                idx = len(self.stoi)
                self.stoi[t] = idx
                self.itos[idx] = t

    def encode(self, text: str, max_length: int = None) -> List[int]:
        toks = text.strip().split()
        ids = [self.stoi.get(t, self.stoi[self.unk_token]) for t in toks]
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int]) -> str:
        words = [self.itos.get(i, self.unk_token) for i in ids]
        # remove special tokens
        words = [w for w in words if w not in (self.pad_token, self.sos_token, self.eos_token)]
        return " ".join(words)
