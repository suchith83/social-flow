# Data ingestion, preprocessing, feature extraction
"""
data_loader.py
---------------
Handles data ingestion, preprocessing, feature engineering for the recommendation pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataLoader:
    """
    Loads user–item interaction data, preprocesses it into numerical form,
    and prepares train/validation/test splits.
    """

    def __init__(self, interactions_path: str, metadata_path: str = None, test_size: float = 0.1, val_size: float = 0.1):
        self.interactions_path = interactions_path
        self.metadata_path = metadata_path
        self.test_size = test_size
        self.val_size = val_size

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def load_data(self) -> pd.DataFrame:
        """Loads interactions from CSV/Parquet and applies encoders."""
        df = pd.read_csv(self.interactions_path)
        if "user_id" not in df or "item_id" not in df or "timestamp" not in df:
            raise ValueError("Interactions file must contain [user_id, item_id, timestamp].")

        # Encode categorical IDs
        df["user_id"] = self.user_encoder.fit_transform(df["user_id"])
        df["item_id"] = self.item_encoder.fit_transform(df["item_id"])

        # Sort by time for temporal split
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits data into train/val/test sets."""
        train, test = train_test_split(df, test_size=self.test_size, shuffle=False)
        train, val = train_test_split(train, test_size=self.val_size, shuffle=False)
        return train, val, test

    def get_item_metadata(self) -> pd.DataFrame:
        """Optional: load item metadata (genres, categories, etc.)."""
        if not self.metadata_path:
            return pd.DataFrame()
        meta = pd.read_csv(self.metadata_path)
        return meta


if __name__ == "__main__":
    loader = DataLoader("data/interactions.csv", "data/items.csv")
    df = loader.load_data()
    train, val, test = loader.train_val_test_split(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
