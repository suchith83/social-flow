"""
Apply SQLite migrations and development seeds to .data/dev.db

Usage:
  python scripts/migrate_and_seed.py
"""
from pathlib import Path
import sqlite3
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "data" / "migrations" / "sqlite"
SEEDS_DIR = ROOT / "data" / "seeds" / "development"
DB_PATH = ROOT / ".data" / "dev.db"


def apply_sql_file(conn: sqlite3.Connection, path: Path):
    sql = path.read_text(encoding="utf-8")
    with conn:
        conn.executescript(sql)


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    print("Applying migrations from:", MIGRATIONS_DIR)
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        print(" -", f.name)
        apply_sql_file(conn, f)

    print("Applying seeds from:", SEEDS_DIR)
    for s in sorted(SEEDS_DIR.glob("*.sql")):
        print(" -", s.name)
        apply_sql_file(conn, s)

    conn.close()
    print("Migrations and seeds applied to:", DB_PATH)


if __name__ == "__main__":
    main()
