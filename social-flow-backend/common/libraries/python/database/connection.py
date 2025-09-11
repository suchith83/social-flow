# common/libraries/python/database/connection.py
"""
Database connection manager with pooling.
Supports PostgreSQL, MySQL, SQLite.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from .config import DatabaseConfig

class ConnectionManager:
    def __init__(self):
        self.engine = create_engine(
            DatabaseConfig.DB_URL,
            pool_size=DatabaseConfig.POOL_SIZE,
            echo=DatabaseConfig.ECHO_SQL,
            future=True,
        )
        self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))

    def get_session(self):
        """Return a new session."""
        return self.SessionLocal()

    def close(self):
        """Dispose engine and release connections."""
        self.engine.dispose()

# Singleton connection manager
db_connection = ConnectionManager()
