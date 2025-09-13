# Data cleaning, feature engineering, enrichment
# transform.py
import pandas as pd
from typing import Dict, Any, Callable
from .utils import logger, timed

class Transformer:
    """
    Applies cleaning, feature engineering, and enrichment to data.
    """

    def __init__(self, transformations: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = None):
        self.transformations = transformations or {}

    @timed
    def clean_nulls(self, df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
        if strategy == "drop":
            return df.dropna()
        elif strategy == "fill_zero":
            return df.fillna(0)
        elif strategy == "ffill":
            return df.ffill()
        else:
            raise ValueError("Invalid null handling strategy")

    @timed
    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    @timed
    def apply_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        for name, fn in self.transformations.items():
            logger.info(f"Applying transformation: {name}")
            df = fn(df)
        return df
