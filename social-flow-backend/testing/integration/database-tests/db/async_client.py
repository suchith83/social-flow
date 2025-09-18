"""
Async DB client using SQLAlchemy async engine with asyncpg.

- Useful for async code path testing.
- Provides an async session factory.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config import settings
from sqlalchemy.engine.url import URL
from typing import AsyncGenerator

DB_VENDOR = settings.db_vendor.lower()

def _build_async_url() -> str:
    if DB_VENDOR in ("postgres", "cockroach"):
        return str(URL.create(
            drivername="postgresql+asyncpg",
            username=settings.db_user,
            password=settings.db_password or None,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name
        ))
    else:
        raise RuntimeError(f"Unsupported DB vendor: {DB_VENDOR}")

ASYNC_DB_URL = _build_async_url()

async_engine = create_async_engine(ASYNC_DB_URL, pool_pre_ping=True, future=True)

AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False, future=True)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
