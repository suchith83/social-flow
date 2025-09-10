"""
Utility functions for trending recommender.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("trending_recommender")


def filter_recent(df: pd.DataFrame, time_col="timestamp", window=timedelta(days=1)):
    """
    Filter interactions within the given time window.
    """
    cutoff = datetime.now() - window
    return df[df[time_col] >= cutoff]
