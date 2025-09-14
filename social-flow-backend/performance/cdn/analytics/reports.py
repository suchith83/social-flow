# Generates dashboards/reports/alerts
# performance/cdn/analytics/reports.py
"""
Reports Module
==============
Generates analytics reports and dashboards.
Supports:
- Latency distribution
- Error rate trends
- Cache hit analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
from .utils import logger

class CDNReports:
    def __init__(self, data: List[Dict]):
        self.df = pd.DataFrame(data)

    def latency_distribution(self, output_file: str = "latency.png"):
        """Plot and save latency distribution."""
        plt.figure(figsize=(8, 5))
        self.df["latency_ms"].dropna().hist(bins=30)
        plt.title("Latency Distribution (ms)")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Frequency")
        plt.savefig(output_file)
        logger.info(f"Saved latency distribution plot to {output_file}")

    def error_rate_over_time(self, output_file: str = "errors.png"):
        """Plot error rate over time."""
        if "timestamp" not in self.df or "status_code" not in self.df:
            return
        self.df["is_error"] = self.df["status_code"] >= 500
        error_series = self.df.groupby("timestamp")["is_error"].mean()
        plt.figure(figsize=(10, 5))
        error_series.plot()
        plt.title("Error Rate Over Time")
        plt.ylabel("Error Rate")
        plt.savefig(output_file)
        logger.info(f"Saved error rate trend plot to {output_file}")
