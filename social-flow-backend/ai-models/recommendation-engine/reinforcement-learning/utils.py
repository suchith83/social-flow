"""
Utility functions for RL recommender.
"""

import torch
import random
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("rl_recommender")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device="cuda"):
    return torch.device(device if torch.cuda.is_available() else "cpu")
