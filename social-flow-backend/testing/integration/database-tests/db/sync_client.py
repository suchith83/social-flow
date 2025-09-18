"""
Synchronous SQLAlchemy engine & session factory for tests.

- Creates a SQLAlchemy Engine connected to the DB from config.
- Provides convenience functions to create/drop test schema or tables.
- Use sessionmaker for transactional tests.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator
from config import settings
from sqlalchemy.engine.url import URL

DB_VENDOR = settings.db_vendor.lower()

def _build_sync_url() -> str:
    # Build standard SQLAlchemy URL
    if DB_VENDOR in ("postgres", "cockroach"):
        return str(URL.create(
            drivername="postgresql+psycopg2",
            username=settings.db_user,
            password=settings.db_password or None,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name
        ))
    else:
        raise RuntimeError(f"Unsupported DB vendor: {DB_VENDOR}")

SYNC_DB_URL = _build_sync_url()

# Create engine with sensible test-time settings
engine = create_engine(
    SYNC_DB_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    future=True,  # SQLAlchemy 1.4 style
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_session() -> Generator:
    """Yield a DB session for tests; tests should roll back as needed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
