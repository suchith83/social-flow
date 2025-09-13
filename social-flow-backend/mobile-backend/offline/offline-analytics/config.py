# Centralized configuration for analytics jobs
"""
Configuration for offline analytics subsystem.
"""

import os
from functools import lru_cache
from pydantic import BaseSettings


class OfflineAnalyticsConfig(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv("ANALYTICS_DATABASE_URL", "sqlite:///./offline_analytics.db")

    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    CELERY_MAX_RETRIES: int = int(os.getenv("ANALYTICS_CELERY_MAX_RETRIES", "3"))

    # Processing
    PROCESS_BATCH_SIZE: int = int(os.getenv("ANALYTICS_PROCESS_BATCH_SIZE", "500"))
    AGGREGATION_INTERVAL_SECONDS: int = int(os.getenv("ANALYTICS_AGG_INTERVAL", "300"))  # 5 minutes

    # Retention (days)
    DATA_RETENTION_DAYS: int = int(os.getenv("ANALYTICS_RETENTION_DAYS", "90"))

    # Export
    EXPORT_PATH: str = os.getenv("ANALYTICS_EXPORT_PATH", "/tmp/analytics_exports")

    # Privacy
    ANONYMIZE_PII: bool = os.getenv("ANALYTICS_ANONYMIZE_PII", "true").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("ANALYTICS_LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_config() -> OfflineAnalyticsConfig:
    return OfflineAnalyticsConfig()
