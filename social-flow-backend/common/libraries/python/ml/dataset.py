# common/libraries/python/ml/dataset.py
"""
Dataset utilities: loading, splitting, batching.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import MLConfig

def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(MLConfig.DATA_DIR, name)
    return pd.read_csv(path)

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, stratify_col: str = None):
    y = df[stratify_col] if stratify_col else None
    train, test = train_test_split(df, test_size=test_size, stratify=y)
    return train, test
