# common/libraries/python/database/database_service.py
"""
Main DatabaseService abstraction.
Combines connection, query builder, ORM, cache.
"""

from typing import Any, Optional
from .connection import db_connection
from .query_builder import QueryBuilder
from .cache import cache_backend

class DatabaseService:
    def __init__(self):
        self.conn = db_connection

    def fetch_one(self, table: str, where: str, cache_key: Optional[str] = None) -> Optional[Any]:
        if cache_key:
            cached = cache_backend.get(cache_key)
            if cached:
                return cached

        session = self.conn.get_session()
        try:
            stmt = QueryBuilder.select(table, where=where)
            row = session.execute(stmt).fetchone()
            if row and cache_key:
                cache_backend.set(cache_key, row._asdict())
            return row._asdict() if row else None
        finally:
            session.close()

    def insert(self, table: str, values: dict):
        session = self.conn.get_session()
        try:
            stmt = QueryBuilder.insert(table, values)
            session.execute(stmt)
            session.commit()
        finally:
            session.close()

    def update(self, table: str, values: dict, where: str):
        session = self.conn.get_session()
        try:
            stmt = QueryBuilder.update(table, values, where)
            session.execute(stmt)
            session.commit()
        finally:
            session.close()

    def delete(self, table: str, where: str):
        session = self.conn.get_session()
        try:
            stmt = QueryBuilder.delete(table, where)
            session.execute(stmt)
            session.commit()
        finally:
            session.close()

database_service = DatabaseService()
