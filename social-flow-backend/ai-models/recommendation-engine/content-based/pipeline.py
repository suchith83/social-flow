"""
Pipeline for training and recommending.
"""

import pandas as pd
from .recommender import ContentBasedRecommender
from .utils import train_test_split_data


class ContentBasedPipeline:
    def __init__(self, method="tfidf", similarity="cosine"):
        self.recommender = ContentBasedRecommender(method=method, similarity=similarity)

    def train(self, df: pd.DataFrame, text_column="description"):
        train, test = train_test_split_data(df)
        self.recommender.fit(train, text_column=text_column)
        return train, test

    def recommend(self, query: str, top_k=10):
        return self.recommender.recommend(query, top_k=top_k)
