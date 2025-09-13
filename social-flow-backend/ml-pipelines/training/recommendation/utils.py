# Shared utilities, logging, configs
"""
utils.py
--------
Shared utility functions: logging, config management, reproducibility.
"""

import os
import random
import numpy as np
import torch
import logging


def set_seed(seed=42):
    """Ensures reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file="training.log"):
    """Configures logging."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("recommendation-pipeline")


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
