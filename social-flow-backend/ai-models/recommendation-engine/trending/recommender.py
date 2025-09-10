"""
Trending Recommender System.
"""

import pandas as pd
from .aggregator import InteractionAggregator
from .scorer import TrendingScorer
from .config import TOP_K


class TrendingRecommender:
    def __init__(self, method="count", time_col="timestamp", item_col="item_id"):
        self.method = method
        self.aggregator = InteractionAggregator(time_col=time_col, item_col=item_col)
        self.scorer = TrendingScorer(time_col=time_col, item_col=item_col)

    def recommend(self, interactions: pd.DataFrame, top_k=TOP_K) -> pd.DataFrame:
        """
        Recommend top trending items.
        """
        if self.method == "count":
            aggregated = self.aggregator.aggregate(interactions)
            return aggregated.sort_values("interaction_count", ascending=False).head(top_k)
        elif self.method == "decay":
            scored = self.scorer.score(interactions)
            return scored.head(top_k)
        else:
            raise ValueError(f"Unsupported trending method: {self.method}")
