# Data extraction (DBs, APIs, files, streaming)
# extract.py
import pandas as pd
import sqlalchemy
import requests
from typing import Dict, Any
from .utils import logger, timed

class Extractor:
    """
    Handles data extraction from multiple sources:
    - SQL databases
    - REST APIs
    - CSV/Parquet files
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @timed
    def from_sql(self, query: str, connection_uri: str) -> pd.DataFrame:
        engine = sqlalchemy.create_engine(connection_uri)
        df = pd.read_sql(query, engine)
        logger.info(f"Extracted {len(df)} rows from SQL database")
        return df

    @timed
    def from_api(self, endpoint: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.json_normalize(data)
        logger.info(f"Extracted {len(df)} rows from API {endpoint}")
        return df

    @timed
    def from_file(self, path: str) -> pd.DataFrame:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format")
        logger.info(f"Extracted {len(df)} rows from file {path}")
        return df
