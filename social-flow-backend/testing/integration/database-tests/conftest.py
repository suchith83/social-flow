"""
Pytest fixtures orchestration:

- Spins up DB test schema (if requested)
- Provides sync and async sessions
- Handles test-level cleanup via rollback or savepoints
- Optionally creates/drops tables before/after the test session
"""

import pytest
from config import settings
from db.migrations import create_all_tables, drop_all_tables
from db.sync_client import engine, SessionLocal, get_session
from db.async_client import async_engine, AsyncSessionLocal, get_async_session
from db.models import Base
from sqlalchemy import text
import asyncio

@pytest.fixture(scope="session", autouse=True)
def prepare_db():
    """
    Session-level fixture to ensure tables are present before running tests.
    This will create tables and drop them after the entire test session to keep DB clean.
    """
    # Create tables
    create_all_tables()
    yield
    # Teardown: drop test tables
    drop_all_tables()

@pytest.fixture()
def db_session():
    """
    Function-scoped SQLAlchemy sync session with rollback semantics.
    Each test runs inside a transaction that is rolled back to keep a clean DB.
    """
    connection = engine.connect()
    trans = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()

@pytest.fixture()
async def db_async_session():
    """
    Function-scoped async session with rollback semantics.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # ensure metadata present
    async with AsyncSessionLocal() as session:
        async with session.begin():
            yield session
        # Rolling back is automatic when the context exits (session.begin()) but ensure cleanup
        await session.rollback()
