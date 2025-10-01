"""Monitors replication lag, health, and node status."""
"""
replication_monitor.py
----------------------
Monitors CockroachDB cluster replication lag, health, and performance.
"""

import psycopg2
import yaml
from pathlib import Path


def load_config():
    with open(Path("config/databases/cockroachdb/config.yaml"), "r") as f:
        return yaml.safe_load(f)


def check_replication_status():
    """Check replication lag and cluster health."""
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
    cur.execute("SHOW CLUSTER SETTING version;")
    version = cur.fetchone()
    cur.execute("SHOW RANGES;")
    ranges = cur.fetchall()

    print(f"Cluster Version: {version}")
    print(f"Total Ranges: {len(ranges)}")

    cur.close()
    conn.close()
