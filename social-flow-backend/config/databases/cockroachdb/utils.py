"""Helper utilities for database diagnostics and debugging."""
"""
utils.py
--------
Helper utilities for CockroachDB debugging, diagnostics, and schema checks.
"""

import psycopg2
import yaml
from pathlib import Path


def load_config():
    with open(Path("config/databases/cockroachdb/config.yaml"), "r") as f:
        return yaml.safe_load(f)


def list_tables():
    """List all tables in the current CockroachDB database."""
    conf = load_config()["database"]
    node = conf["nodes"][0]
    conn = psycopg2.connect(
        dbname=conf["name"],
        user=conf["user"],
        password=conf["password"],
        host=node["host"],
        port=node["port"],
        sslmode="require"
    )
    cur = conn.cursor()
    cur.execute("SHOW TABLES;")
    tables = cur.fetchall()
    print("Tables:", tables)
    cur.close()
    conn.close()
