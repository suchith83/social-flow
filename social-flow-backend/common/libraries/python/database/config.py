# common/libraries/python/database/config.py
"""
Database configuration loader.
Supports environment variable overrides.
"""

import os

class DatabaseConfig:
    DB_URL = os.getenv("DB_URL", "sqlite:///./local.db")  # default SQLite
    POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    ECHO_SQL = os.getenv("DB_ECHO_SQL", "false").lower() == "true"

    # Migration
    MIGRATIONS_DIR = os.getenv("DB_MIGRATIONS_DIR", "./migrations")

    # Cache
    CACHE_BACKEND = os.getenv("DB_CACHE_BACKEND", "memory")  # memory | redis
    CACHE_TTL = int(os.getenv("DB_CACHE_TTL", "300"))  # 5 min
