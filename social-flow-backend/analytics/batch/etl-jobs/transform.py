import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim
from .utils import logger, hash_record


class Transformer:
    """Handles data transformation"""

    def __init__(self):
        self.spark = SparkSession.builder.appName("ETLTransformer").getOrCreate()

    def clean_user_data(self, records):
        """Normalize user data with pandas"""
        df = pd.DataFrame(records)
        df.drop_duplicates(inplace=True)
        df.fillna({"username": "unknown"}, inplace=True)
        df["username"] = df["username"].str.lower().str.strip()
        logger.info(f"Cleaned user dataset: {len(df)} records")
        return df.to_dict(orient="records")

    def enrich_with_hash(self, records):
        """Add hash column for deduplication"""
        for record in records:
            record["record_hash"] = hash_record(record)
        logger.info("Enriched records with hashes")
        return records

    def transform_videos_spark(self, records):
        """Use Spark for video data transformation"""
        df = self.spark.createDataFrame(records)
        df = df.withColumn("title", trim(lower(col("title"))))
        logger.info(f"Transformed {df.count()} video records with Spark")
        return df.toPandas().to_dict(orient="records")
