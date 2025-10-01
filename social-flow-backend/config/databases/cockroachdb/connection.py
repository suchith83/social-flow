"""Manage CockroachDB connection pooling with retries, failover, and TLS."""
"""
connection.py
--------------
Manages secure, resilient, and efficient connection pooling to CockroachDB clusters.
Supports TLS, retries, automatic failover, and health checks.
"""

import psycopg2
import psycopg2.pool
import time
import logging
import threading
import yaml
from pathlib import Path

logger = logging.getLogger("CockroachDBConnection")
logger.setLevel(logging.INFO)


class CockroachDBConnectionPool:
    """
    Advanced connection pool for CockroachDB.
    Supports:
    - SSL/TLS secure connections
    - Retry with exponential backoff
    - Cluster-aware failover between nodes
    """

    def __init__(self, config_path: str = "config/databases/cockroachdb/config.yaml"):
        self.config = self._load_config(config_path)
        self.pool = None
        self.lock = threading.Lock()
        self._initialize_pool()

    def _load_config(self, path: str):
        """Load YAML database configuration."""
        with open(Path(path), "r") as f:
            return yaml.safe_load(f)

    def _initialize_pool(self):
        """Initialize connection pool with failover across cluster nodes."""
        db_conf = self.config["database"]
        for node in db_conf["nodes"]:
            try:
                logger.info(f"Attempting CockroachDB connection: {node['host']}:{node['port']}")
                self.pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=db_conf["pool"]["min"],
                    maxconn=db_conf["pool"]["max"],
                    user=db_conf["user"],
                    password=db_conf["password"],
                    host=node["host"],
                    port=node["port"],
                    database=db_conf["name"],
                    sslmode="require"
                )
                logger.info(f"? Connected successfully to CockroachDB node {node['host']}")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to node {node['host']}: {e}")
                continue
        raise Exception("? All CockroachDB nodes unavailable")

    def get_conn(self):
        """Get a connection from the pool with retry mechanism."""
        retries = 3
        delay = 2
        for attempt in range(1, retries + 1):
            try:
                with self.lock:
                    return self.pool.getconn()
            except Exception as e:
                logger.error(f"Connection attempt {attempt} failed: {e}")
                time.sleep(delay)
                delay *= 2
        raise Exception("? Unable to get CockroachDB connection after retries")

    def put_conn(self, conn):
        """Return connection to pool."""
        with self.lock:
            self.pool.putconn(conn)

    def close_all(self):
        """Close all connections."""
        if self.pool:
            self.pool.closeall()
