"""
Utility functions for preprocessing, logging, and common helpers.
"""

import re
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("content_based_recommender")


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercasing, removing special chars, numbers.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Split dataframe into train and test sets.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
