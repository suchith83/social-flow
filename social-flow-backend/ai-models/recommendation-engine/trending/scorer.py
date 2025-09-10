"""
Score items based on trending signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from .config import DECAY_FACTOR


class TrendingScorer:
    def __init__(self, time_col="timestamp", item_col="item_id"):
        self.time_col = time_col
        self.item_col = item_col

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply decay-based trending score.
        More recent interactions weigh higher.
        """
        now = datetime.now()
        df["age_hours"] = (now - df[self.time_col]).dt.total_seconds() / 3600
        df["weight"] = np.power(DECAY_FACTOR, df["age_hours"] / 24)  # decay per day
        df["score"] = df["weight"]
        scores = df.groupby(self.item_col)["score"].sum().reset_index()
        scores = scores.sort_values("score", ascending=False)
        return scores
