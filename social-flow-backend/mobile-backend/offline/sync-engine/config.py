# Centralized configuration (retry, backoff, scheduling)
"""
Configuration for the sync engine.
"""

import os
from functools import lru_cache
from pydantic import BaseSettings


class SyncConfig(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv("SYNC_DATABASE_URL", "sqlite:///./sync_engine.db")

    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # Sync behaviour
    MAX_PUSH_BATCH_SIZE: int = int(os.getenv("SYNC_MAX_PUSH_BATCH", "200"))
    MAX_PULL_PAGE: int = int(os.getenv("SYNC_MAX_PULL_PAGE", "500"))

    # Tombstone retention (seconds)
    TOMBSTONE_RETENTION_SECONDS: int = int(os.getenv("SYNC_TOMBSTONE_RETENTION_SECONDS", 7 * 24 * 3600))

    # Conflict resolution default
    DEFAULT_CONFLICT_STRATEGY: str = os.getenv("SYNC_DEFAULT_CONFLICT_STRATEGY", "lww")  # last-write-wins

    # Misc
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_config() -> SyncConfig:
    return SyncConfig()
