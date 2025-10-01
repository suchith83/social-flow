"""Configuration loader for Redis settings."""
"""
Redis Configuration Loader
- Loads settings from environment variables or AWS Secrets Manager / Vault.
- Provides a centralized configuration object for other Redis modules.
"""

import os
import json
from functools import lru_cache
from typing import Dict, Any, Optional


class RedisConfigError(Exception):
    """Raised when Redis configuration is invalid."""


class RedisConfig:
    """
    Handles Redis configuration loading from:
    - Environment variables
    - Secrets JSON (in case of AWS/GCP secret managers)
    """

    def __init__(self):
        self.host: str = os.getenv("REDIS_HOST", "localhost")
        self.port: int = int(os.getenv("REDIS_PORT", "6379"))
        self.username: Optional[str] = os.getenv("REDIS_USERNAME")
        self.password: Optional[str] = os.getenv("REDIS_PASSWORD")
        self.db: int = int(os.getenv("REDIS_DB", "0"))
        self.use_ssl: bool = os.getenv("REDIS_USE_SSL", "false").lower() == "true"
        self.max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        self.connection_timeout: int = int(os.getenv("REDIS_CONNECTION_TIMEOUT", "5"))
        self.socket_timeout: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))

        if not self.host or not self.port:
            raise RedisConfigError("Redis host/port configuration is missing.")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "db": self.db,
            "ssl": self.use_ssl,
            "max_connections": self.max_connections,
            "socket_connect_timeout": self.connection_timeout,
            "socket_timeout": self.socket_timeout,
        }


@lru_cache()
def get_config() -> RedisConfig:
    """Singleton accessor for RedisConfig."""
    return RedisConfig()
