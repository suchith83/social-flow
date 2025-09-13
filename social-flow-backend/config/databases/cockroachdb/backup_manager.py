"""Automated incremental backups & restore using CockroachDB native features."""
"""
backup_manager.py
-----------------
Automates CockroachDB backups and restores using native BACKUP/RESTORE.
"""

import psycopg2
import yaml
from pathlib import Path
import datetime


def load_config():
    with open(Path("config/databases/cockroachdb/config.yaml"), "r") as f:
        return yaml.safe_load(f)


def run_backup():
    """Perform incremental backup to S3/GCS/Azure."""
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
    backup_path = f"s3://cockroach-backups/{conf['name']}/{datetime.date.today()}"
    cur.execute(f"BACKUP TO '{backup_path}' WITH revision_history;")
    print(f"✅ Backup completed: {backup_path}")
    conn.commit()
    cur.close()
    conn.close()
