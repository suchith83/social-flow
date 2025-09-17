"""
Configuration handler for Analytics Data pipeline.
Uses environment variables and dynamic defaults.
"""

import os
from pydantic import BaseSettings


class AnalyticsConfig(BaseSettings):
    ENV: str = os.getenv("ENV", "development")
    KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DB_URL: str = os.getenv("DB_URL", "postgresql://user:pass@localhost:5432/analytics")
    CLICKHOUSE_URL: str = os.getenv("CLICKHOUSE_URL", "http://localhost:8123")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "500"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = AnalyticsConfig()
