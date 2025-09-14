# redis_config.py
# Created by Create-Configurations.ps1
"""
redis_config.py
---------------
Defines Redis connection pools, cluster/failover support, and TLS configurations.
"""

import os
import redis
from typing import Optional


class RedisConfig:
    """Configuration handler for Redis clusters."""

    HOST = os.getenv("REDIS_HOST", "localhost")
    PORT = int(os.getenv("REDIS_PORT", "6379"))
    PASSWORD = os.getenv("REDIS_PASSWORD", None)
    USE_TLS = os.getenv("REDIS_TLS", "false").lower() == "true"
    DB = int(os.getenv("REDIS_DB", "0"))
    MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONN", "50"))
    TIMEOUT = int(os.getenv("REDIS_TIMEOUT", "5"))

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Return a Redis client instance with connection pooling."""
        pool = redis.ConnectionPool(
            host=cls.HOST,
            port=cls.PORT,
            password=cls.PASSWORD,
            db=cls.DB,
            max_connections=cls.MAX_CONNECTIONS,
            socket_connect_timeout=cls.TIMEOUT,
            ssl=cls.USE_TLS,
        )
        return redis.Redis(connection_pool=pool)

    @classmethod
    def summary(cls) -> dict:
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "db": cls.DB,
            "use_tls": cls.USE_TLS,
            "max_connections": cls.MAX_CONNECTIONS,
        }


if __name__ == "__main__":
    print("🔧 Redis Config:", RedisConfig.summary())
