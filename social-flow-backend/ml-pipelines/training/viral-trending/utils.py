# Logging, configs, reproducibility
"""
utils.py
--------
Utility functions: logging, reproducibility.
"""

import numpy as np
import random
import torch
import logging


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file="viral_training.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("viral-trending")
