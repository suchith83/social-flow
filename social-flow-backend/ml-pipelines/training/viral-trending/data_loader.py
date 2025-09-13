# Load posts, engagement logs, temporal features
"""
data_loader.py
--------------
Loads engagement + content data for viral trending prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


class ViralDataLoader:
    def __init__(self, posts_path: str, engagements_path: str, test_size: float = 0.1, val_size: float = 0.1):
        self.posts_path = posts_path
        self.engagements_path = engagements_path
        self.test_size = test_size
        self.val_size = val_size

    def load(self) -> pd.DataFrame:
        posts = pd.read_csv(self.posts_path)
        engagements = pd.read_csv(self.engagements_path)

        # Aggregate engagements (likes, shares, views, comments) at post level
        agg = engagements.groupby("post_id").agg(
            views=("view", "sum"),
            likes=("like", "sum"),
            shares=("share", "sum"),
            comments=("comment", "sum"),
            unique_users=("user_id", "nunique"),
        ).reset_index()

        df = posts.merge(agg, on="post_id", how="left").fillna(0)

        # Label viral posts (e.g., top 5% by engagement)
        df["viral"] = (df["views"] > np.percentile(df["views"], 95)).astype(int)
        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(df, test_size=self.test_size, shuffle=True, random_state=42)
        train, val = train_test_split(train, test_size=self.val_size, shuffle=True, random_state=42)
        return train, val, test


if __name__ == "__main__":
    loader = ViralDataLoader("data/posts.csv", "data/engagements.csv")
    df = loader.load()
    train, val, test = loader.split(df)
    print("Data loaded:", df.shape, "Train/Val/Test:", len(train), len(val), len(test))
