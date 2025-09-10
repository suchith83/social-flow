"""
Aggregates interactions for trending scoring.
"""

import pandas as pd
from datetime import timedelta
from .utils import filter_recent, logger
from .config import TREND_WINDOW


class InteractionAggregator:
    def __init__(self, time_col="timestamp", item_col="item_id"):
        self.time_col = time_col
        self.item_col = item_col

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate interactions by item within trend window.
        """
        df_recent = filter_recent(df, time_col=self.time_col, window=TREND_WINDOW)
        grouped = df_recent.groupby(self.item_col).size().reset_index(name="interaction_count")
        logger.info(f"Aggregated {len(grouped)} trending items in window {TREND_WINDOW}")
        return grouped
