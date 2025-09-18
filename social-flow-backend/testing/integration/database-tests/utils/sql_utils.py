"""
SQL utility helpers used by tests for direct SQL verification and convenience.
"""

from sqlalchemy import text
from db.sync_client import engine

def scalar_query(sql: str, **params):
    with engine.connect() as conn:
        res = conn.execute(text(sql), params)
        row = res.first()
        return row[0] if row else None

def execute_script(sql: str):
    with engine.begin() as conn:
        conn.execute(text(sql))
