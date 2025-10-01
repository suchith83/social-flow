# common/libraries/python/database/database_service.py
"""
Main DatabaseService abstraction.
Combines connection, query builder, ORM, cache.
"""

from typing import Any, Optional, Dict
from .connection import db_connection
from .query_builder import QueryBuilder
from .cache import cache_backend
import sqlite3
import json
import os
import threading
import time

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

    def _get_conn():
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


    def _ensure_schema():
        with _init_lock:
            conn = _get_conn()
            with conn:
                # table for persisted recommendation feedback
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recommendation_feedback (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        item_id TEXT,
                        action TEXT,
                        timestamp INTEGER,
                        payload_json TEXT,
                        created_at REAL
                    );
                    """
                )
                # generic key-value store for other small needs
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kv_store (
                        id TEXT PRIMARY KEY,
                        table_name TEXT,
                        payload_json TEXT,
                        created_at REAL
                    );
                    """
                )
            conn.close()


    # Ensure schema at import time (safe and idempotent)
    try:
        _ensure_schema()
    except Exception:
        # In rare environments file permissions may fail; consumers should handle exceptions on insert.
        pass


    def insert(self, table: str, payload: Dict[str, Any]) -> None:
        """
        Insert payload into named table.
        Special-case `recommendation_feedback` to extract fields for querying.
        Other tables are persisted into kv_store as JSON.
        """
        try:
            _ensure_schema()
            conn = _get_conn()
            with conn:
                ts = float(time.time())
                if table == "recommendation_feedback":
                    # attempt to pull common fields; fall back to storing raw JSON
                    user_id = payload.get("user_id")
                    item_id = payload.get("item_id")
                    action = payload.get("action")
                    timestamp = int(payload.get("timestamp") or payload.get("ts") or ts)
                    row_id = f"fb_{int(ts*1000)}"
                    conn.execute(
                        "INSERT INTO recommendation_feedback (id,user_id,item_id,action,timestamp,payload_json,created_at) VALUES (?,?,?,?,?,?,?)",
                        (row_id, user_id, item_id, action, timestamp, json.dumps(payload), ts),
                    )
                else:
                    row_id = f"kv_{int(ts*1000)}"
                    conn.execute(
                        "INSERT INTO kv_store (id,table_name,payload_json,created_at) VALUES (?,?,?,?)",
                        (row_id, table, json.dumps(payload), ts),
                    )
        finally:
            try:
                conn.close()
            except Exception:
                pass


# Singleton instance expected by importers
database_service = DatabaseService()
