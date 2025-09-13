# Cleans, aggregates, and validates results
# ================================================================
# File: postprocessing.py
# Purpose: Post-process inference results
# ================================================================

import logging
import pandas as pd

logger = logging.getLogger("PostProcessor")


class PostProcessor:
    """
    Handles:
    - Thresholding
    - Business rule enforcement
    - Aggregation
    """

    def __init__(self, config: dict):
        self.config = config

    def process(self, results):
        df = pd.DataFrame(results)
        if self.config.get("threshold"):
            threshold = self.config["threshold"]
            df["flagged"] = df.iloc[:, 0] > threshold
        return df
