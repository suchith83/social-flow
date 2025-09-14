# Identifies trends and patterns over time
# monitoring/logging/analysis/trend_analysis.py
"""
Trend analysis module.
Performs time-series aggregation, rolling averages, and prepares data
for forecasting models.
"""

import pandas as pd
from .config import CONFIG


class TrendAnalyzer:
    def __init__(self):
        self.agg_window = CONFIG["TREND_ANALYSIS"]["aggregation_window"]
        self.rolling_window = CONFIG["TREND_ANALYSIS"]["rolling_window"]

    def aggregate(self, logs: list, metric_field="level"):
        """Aggregate logs into time-series counts by level or other metric."""
        df = pd.DataFrame(logs)
        if "timestamp" not in df:
            return pd.DataFrame()

        df.set_index("timestamp", inplace=True)
        counts = df.groupby([pd.Grouper(freq=self.agg_window), metric_field]).size()
        return counts.unstack(fill_value=0)

    def rolling_average(self, timeseries: pd.DataFrame):
        """Compute rolling average for smoothing trends."""
        return timeseries.rolling(self.rolling_window, min_periods=1).mean()
