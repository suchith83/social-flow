import sqlite3
from typing import Optional, Dict, Any
import os

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    salt TEXT NOT NULL,
    pwd_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL
);
"""

def get_conn(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str) -> None:
    conn = get_conn(db_path)
    with conn:
        conn.executescript(SCHEMA_SQL)
    conn.close()

def create_user(db_path: str, user_id: str, email: str, display_name: str, salt: str, pwd_hash: str, role: str = "user") -> None:
    conn = get_conn(db_path)
    with conn:
        conn.execute(
            "INSERT INTO users (id,email,display_name,salt,pwd_hash,role,created_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, email, display_name, salt, pwd_hash, role, float(__import__("time").time()))
        )
    conn.close()

def get_user_by_email(db_path: str, email: str) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(db_path: str, user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
