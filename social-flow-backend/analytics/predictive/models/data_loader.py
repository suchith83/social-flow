"""
Data loading utilities.
Supports:
 - Direct Snowflake queries (via pandas.read_sql)
 - Local Parquet / feature-store loading
 - Simple sampling and stratified splitting helpers
"""

from typing import Tuple
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .config import settings
from .utils import logger


class DataLoader:
    def __init__(self, snowflake_conn=None):
        """
        snowflake_conn: optional connection object (sqlalchemy engine or connector)
        If None, methods will expect local feature files under settings.FEATURE_STORE_PATH
        """
        self.snowflake_conn = snowflake_conn

    def load_from_parquet(self, path: str) -> pd.DataFrame:
        """Load dataset from parquet file (local or s3 via fsspec path)"""
        logger.info(f"Loading parquet data from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded dataframe with shape {df.shape}")
        return df

    def load_feature_table(self, table_name: str) -> pd.DataFrame:
        """
        Load feature table from feature store path.
        Convention: settings.FEATURE_STORE_PATH/{table_name}.parquet
        """
        path = os.path.join(settings.FEATURE_STORE_PATH, f"{table_name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature table not found: {path}")
        return self.load_from_parquet(path)

    def fetch_from_sql(self, query: str) -> pd.DataFrame:
        """Fetch data by SQL using snowflake_conn (must be a SQLAlchemy engine or connector supported by pd.read_sql)."""
        if self.snowflake_conn is None:
            raise RuntimeError("No snowflake_conn provided for SQL fetch.")
        logger.info("Executing SQL query to fetch data")
        df = pd.read_sql(query, self.snowflake_conn)
        logger.info(f"Fetched {len(df)} rows")
        return df

    def train_test_split(
        self, df: pd.DataFrame, target: str, test_size: float = 0.2, stratify: str | None = None, random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Utility for splitting into train/test sets.
        Returns X_train, X_test, y_train, y_test (dataframes for Xs; series for ys)
        """
        random_state = random_state or settings.RANDOM_SEED
        y = df[target]
        X = df.drop(columns=[target])
        stratify_vals = df[stratify] if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_vals, random_state=random_state
        )
        logger.info(f"Train/test split: {X_train.shape}/{X_test.shape}")
        return X_train, X_test, y_train, y_test
