# Online (Redis) & Offline (Parquet/SQL) storage
# storage.py
import pandas as pd
import sqlalchemy
import redis
import pickle
from typing import Dict, Any
from .utils import logger, timed

class OfflineStore:
    """
    Stores features in SQL/Parquet for batch training.
    """

    def __init__(self, connection_uri: str = None):
        self.connection_uri = connection_uri

    @timed
    def save_to_parquet(self, df: pd.DataFrame, path: str):
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} rows to offline store at {path}")

    @timed
    def save_to_sql(self, df: pd.DataFrame, table: str):
        engine = sqlalchemy.create_engine(self.connection_uri)
        df.to_sql(table, engine, if_exists="replace", index=False)
        logger.info(f"Saved {len(df)} rows to SQL table {table}")

class OnlineStore:
    """
    Stores features in Redis for low-latency serving.
    """

    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    @timed
    def write(self, entity_id: str, features: Dict[str, Any]):
        self.client.set(entity_id, pickle.dumps(features))
        logger.info(f"Stored features for entity_id={entity_id}")

    @timed
    def read(self, entity_id: str) -> Dict[str, Any]:
        data = self.client.get(entity_id)
        return pickle.loads(data) if data else {}
