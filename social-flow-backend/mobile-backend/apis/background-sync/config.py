# Configs: retry intervals, backoff strategies
"""
Configuration settings for Background Sync.
Centralized for flexibility across environments (dev, staging, prod).
"""

import os
from functools import lru_cache


class BackgroundSyncConfig:
    # Celery broker (e.g., Redis, RabbitMQ)
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

    # Celery backend for storing task results
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # Retry policy
    MAX_RETRIES: int = int(os.getenv("SYNC_MAX_RETRIES", 5))
    RETRY_BACKOFF: int = int(os.getenv("SYNC_RETRY_BACKOFF", 60))  # seconds

    # Batch size for sync processing
    BATCH_SIZE: int = int(os.getenv("SYNC_BATCH_SIZE", 100))

    # Database connection
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./background_sync.db")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache()
def get_config() -> BackgroundSyncConfig:
    return BackgroundSyncConfig()
