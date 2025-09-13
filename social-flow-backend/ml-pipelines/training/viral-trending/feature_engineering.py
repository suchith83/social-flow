# Extract content, social, temporal features
"""
feature_engineering.py
----------------------
Extracts content, social, and temporal features for viral prediction.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        # Content features from captions/text
        text_features = self.vectorizer.fit_transform(df["caption"].fillna("")).toarray()

        # Engagement ratios (shares/views, likes/views, comments/views)
        df["like_rate"] = df["likes"] / (df["views"] + 1e-6)
        df["share_rate"] = df["shares"] / (df["views"] + 1e-6)
        df["comment_rate"] = df["comments"] / (df["views"] + 1e-6)

        numeric_features = df[["views", "likes", "shares", "comments", "unique_users", "like_rate", "share_rate", "comment_rate"]].values

        X = np.hstack([text_features, numeric_features])
        X = self.scaler.fit_transform(X)
        y = df["viral"].values
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        text_features = self.vectorizer.transform(df["caption"].fillna("")).toarray()
        df["like_rate"] = df["likes"] / (df["views"] + 1e-6)
        df["share_rate"] = df["shares"] / (df["views"] + 1e-6)
        df["comment_rate"] = df["comments"] / (df["views"] + 1e-6)

        numeric_features = df[["views", "likes", "shares", "comments", "unique_users", "like_rate", "share_rate", "comment_rate"]].values
        X = np.hstack([text_features, numeric_features])
        X = self.scaler.transform(X)
        return X
