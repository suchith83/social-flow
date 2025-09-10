"""
Utility functions for preprocessing, vocabulary handling, and logging.
"""

import torch
import random
import numpy as np
import logging
from collections import Counter
import pickle
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("caption_generation")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Vocabulary:
    """
    Vocabulary for tokenizing captions.
    """

    def __init__(self, freq_threshold=5, max_size=20000):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocab(self, sentence_list):
        freqs = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in sentence.lower().split(" "):
                freqs[word] += 1

        for word, freq in freqs.most_common(self.max_size):
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        return [
            self.stoi.get(word, self.stoi["<UNK>"])
            for word in text.lower().split(" ")
        ]

def save_vocab(vocab, path):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info(f"Vocabulary saved at {path}")

def load_vocab(path):
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    logger.info(f"Vocabulary loaded from {path}")
    return vocab
