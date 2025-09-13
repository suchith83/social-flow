# Configuration: credentials, retry intervals, templates
"""
Configuration for push notifications module.
Centralizes provider credentials, retry/backoff policy, batching, and sandbox toggles.
"""

import os
from functools import lru_cache
from pydantic import BaseSettings


class PushConfig(BaseSettings):
    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./push_notifications.db")

    # Retry / backoff
    MAX_RETRIES: int = int(os.getenv("PUSH_MAX_RETRIES", "5"))
    RETRY_BACKOFF_SECONDS: int = int(os.getenv("PUSH_RETRY_BACKOFF_SECONDS", "30"))

    # Batch sending
    MAX_BATCH_SIZE: int = int(os.getenv("PUSH_MAX_BATCH_SIZE", "100"))

    # Provider toggles / credentials (env var placeholders)
    FCM_SERVER_KEY: str = os.getenv("FCM_SERVER_KEY", "")
    APNS_AUTH_KEY_PATH: str = os.getenv("APNS_AUTH_KEY_PATH", "")
    APNS_KEY_ID: str = os.getenv("APNS_KEY_ID", "")
    APNS_TEAM_ID: str = os.getenv("APNS_TEAM_ID", "")

    # Environment
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Template config
    DEFAULT_SOUND: str = os.getenv("PUSH_DEFAULT_SOUND", "default")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_config() -> PushConfig:
    return PushConfig()
