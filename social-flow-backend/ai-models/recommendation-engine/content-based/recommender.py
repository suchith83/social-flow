"""
Main Content-Based Recommender class.
"""

import numpy as np
import pandas as pd
from .feature_extraction import FeatureExtractor
from .similarity import SimilarityComputer
from .config import TOP_K
from .utils import logger


class ContentBasedRecommender:
    def __init__(self, method="tfidf", similarity="cosine"):
        self.feature_extractor = FeatureExtractor(method=method)
        self.similarity_computer = SimilarityComputer(metric=similarity)
        self.item_features = None
        self.items_df = None

    def fit(self, items_df: pd.DataFrame, text_column: str = "description"):
        """
        Fit recommender on items.
        """
        self.items_df = items_df.reset_index(drop=True)
        self.item_features = self.feature_extractor.fit_transform(items_df[text_column])
        logger.info("Recommender fitted on items")

    def recommend(self, query: str, top_k: int = TOP_K) -> pd.DataFrame:
        """
        Recommend top-k items given a query.
        """
        query_vector = self.feature_extractor.transform(pd.Series([query]))[0]
        scores = self.similarity_computer.compute(self.item_features, query_vector)

        top_indices = np.argsort(scores)[::-1][:top_k]
        recommendations = self.items_df.iloc[top_indices].copy()
        recommendations["similarity_score"] = scores[top_indices]
        return recommendations
