# Clean and preprocess user/content data
# ============================
# File: preprocessing.py
# ============================
import pandas as pd
from typing import Tuple, Dict

def encode_ids(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    mapping = {k: i for i, k in enumerate(df[col].unique())}
    df[col] = df[col].map(mapping)
    return df, mapping
