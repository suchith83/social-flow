"""
Simple migrations helper for tests.

This provides helper functions to create/drop the testing tables directly via SQLAlchemy
without requiring a real Alembic environment in the test. For production migrations, use Alembic.

We still include an `alembic/env.py` skeleton for compatibility.
"""

from db.models import Base
from db.sync_client import engine
from sqlalchemy import text

def create_all_tables():
    """Create all models' tables in the current DB (honors metadata)."""
    Base.metadata.create_all(bind=engine)

def drop_all_tables():
    """Drop all tables defined in models (best-effort)."""
    Base.metadata.drop_all(bind=engine)

def table_exists(table_name: str) -> bool:
    with engine.connect() as conn:
        res = conn.execute(text(f"SELECT to_regclass('{table_name}')"))
        return res.scalar() is not None
