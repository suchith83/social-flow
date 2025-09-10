"""
Utility functions for preprocessing, logging, and helpers.
"""

import torch
import numpy as np
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("deep_learning_recommender")


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "cuda"):
    """
    Return proper device (GPU if available, else CPU).
    """
    return torch.device(device if torch.cuda.is_available() else "cpu")
