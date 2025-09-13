# Loading to DBs, warehouses, object storage
# load.py
import pandas as pd
import sqlalchemy
import boto3
from typing import Dict, Any
from .utils import logger, timed

class Loader:
    """
    Loads transformed data into various targets:
    - SQL database
    - Data warehouse (Snowflake, Redshift, etc.)
    - Object storage (S3)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @timed
    def to_sql(self, df: pd.DataFrame, table: str, connection_uri: str, if_exists="replace"):
        engine = sqlalchemy.create_engine(connection_uri)
        df.to_sql(table, engine, if_exists=if_exists, index=False)
        logger.info(f"Loaded {len(df)} rows into SQL table {table}")

    @timed
    def to_s3(self, df: pd.DataFrame, bucket: str, key: str, format: str = "parquet"):
        s3 = boto3.client("s3")
        if format == "csv":
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
        elif format == "parquet":
            import io
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"Loaded {len(df)} rows to S3://{bucket}/{key}")
