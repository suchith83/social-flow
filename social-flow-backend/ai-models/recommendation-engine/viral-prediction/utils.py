"""
Utility functions: logging, seed, time helpers, persistence
"""

import logging
import random
import numpy as np
import os
import pickle
from .config import RANDOM_SEED, MODEL_DIR
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("viral_prediction")

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def save_pickle(obj, name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved {name} to {path}")
    return path

def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded {name} from {path}")
    return obj

def now_iso():
    return datetime.utcnow().isoformat()
