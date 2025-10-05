"""
Database configuration and connection management.

This module handles database connections, session management, and
database initialization for the Social Flow backend.
 
Key change: the async engine and sessionmaker are initialized lazily.
This prevents importing or initializing database drivers (e.g., asyncpg)
at import time, which is important for tests that use SQLite/aiosqlite.
"""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)

# Database metadata
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# Create declarative base (SQLAlchemy 2.0 style)
class Base(DeclarativeBase):
    metadata = metadata

# Lazily initialized async engine & session maker
_engine: Optional[AsyncEngine] = None
_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def _build_engine() -> AsyncEngine:
    """Construct the SQLAlchemy AsyncEngine based on settings.

    - In TESTING mode, force SQLite (aiosqlite) to avoid asyncpg dependency.
    - In non-testing mode, use DATABASE_URL from settings.
    - Apply conservative engine kwargs compatible across drivers.
    """
    # Determine URL
    if settings.TESTING:
        db_url = "sqlite+aiosqlite:///./test.db"
    else:
        db_url = str(settings.DATABASE_URL)

    # Engine kwargs
    engine_kwargs = {
        "echo": settings.DEBUG,
        "future": True,
    }

    # Avoid pooling params that may conflict with SQLite
    # Add pre-ping for non-sqlite drivers
    if not db_url.startswith("sqlite+"):
        engine_kwargs.update({
            "pool_pre_ping": True,
            "pool_recycle": 300,
        })

    return create_async_engine(db_url, **engine_kwargs)


def get_engine() -> AsyncEngine:
    """Get or create the global AsyncEngine instance."""
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get or create the global async sessionmaker instance."""
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_maker


# Backwards-compatible alias used by some modules (initialized lazily on first use)
async def async_session_maker() -> async_sessionmaker[AsyncSession]:
    return get_session_maker()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    # Create session from lazily initialized sessionmaker
    SessionLocal = get_session_maker()
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    try:
        async with get_engine().begin() as conn:
            # Import models package to ensure all model classes are registered
            import app.models  # noqa: F401
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.warning("Application will continue but database functionality may be limited")
        # Don't raise - allow application to start even if DB init fails


async def close_db() -> None:
    """Close database connections."""
    try:
        eng = get_engine()
        await eng.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")
        raise
