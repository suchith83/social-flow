# Efficient batch data loading (from DB, cloud, etc.)
# ================================================================
# File: data_loader.py
# Purpose: Handles loading batch data from multiple sources
# ================================================================

import logging
import pandas as pd
import sqlalchemy
from utils import retry

logger = logging.getLogger("DataLoader")


class BatchDataLoader:
    """
    Supports loading batch inference data from:
    - Databases (SQLAlchemy)
    - CSV / Parquet files
    - Cloud storage buckets
    """

    def __init__(self, config: dict):
        self.config = config

    @retry(max_attempts=3, delay=3, exceptions=(Exception,))
    def load(self):
        source_type = self.config.get("source", "csv")
        if source_type == "csv":
            return pd.read_csv(self.config["path"])
        elif source_type == "parquet":
            return pd.read_parquet(self.config["path"])
        elif source_type == "db":
            engine = sqlalchemy.create_engine(self.config["connection"])
            query = self.config["query"]
            return pd.read_sql(query, engine)
        else:
            raise ValueError(f"Unsupported data source: {source_type}")
