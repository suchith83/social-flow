# Centralized configuration (cache size, TTL, backends)
"""
Configuration for content caching subsystem.
Centralizes Redis, disk cache directories, TTLs, prefetch settings, and Celery config.
"""

import os
from functools import lru_cache
from pydantic import BaseSettings


class ContentCacheConfig(BaseSettings):
    # Storage
    REDIS_URL: str = os.getenv("CACHE_REDIS_URL", "redis://localhost:6379/2")
    DISK_CACHE_PATH: str = os.getenv("DISK_CACHE_PATH", "/tmp/content_cache")
    MAX_DISK_USAGE_BYTES: int = int(os.getenv("CACHE_MAX_DISK_USAGE_BYTES", 5 * 1024 * 1024 * 1024))  # 5GB

    # TTL settings (seconds)
    DEFAULT_TTL: int = int(os.getenv("CACHE_DEFAULT_TTL", 24 * 3600))  # 24 hours
    STALE_THRESHOLD: int = int(os.getenv("CACHE_STALE_THRESHOLD", 12 * 3600))  # 12 hours

    # Prefetching
    PREFETCH_BATCH_SIZE: int = int(os.getenv("CACHE_PREFETCH_BATCH_SIZE", 50))
    PREFETCH_CRON: str = os.getenv("CACHE_PREFETCH_CRON", "0 * * * *")  # example: every hour

    # Eviction
    MIN_FREE_DISK_BYTES: int = int(os.getenv("CACHE_MIN_FREE_DISK_BYTES", 1 * 1024 * 1024 * 1024))  # 1GB

    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    CELERY_TASK_MAX_RETRIES: int = int(os.getenv("CACHE_TASK_MAX_RETRIES", 3))

    LOG_LEVEL: str = os.getenv("CACHE_LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_config() -> ContentCacheConfig:
    return ContentCacheConfig()
