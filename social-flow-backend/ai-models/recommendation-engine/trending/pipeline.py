"""
Pipeline for trending recommendation engine.
"""

import pandas as pd
from .recommender import TrendingRecommender


class TrendingPipeline:
    def __init__(self, method="count"):
        self.recommender = TrendingRecommender(method=method)

    def run(self, interactions: pd.DataFrame, top_k=10):
        return self.recommender.recommend(interactions, top_k=top_k)
