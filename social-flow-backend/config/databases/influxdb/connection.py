"""Secure connection to InfluxDB v2 (token auth, retries, failover)."""
"""
connection.py
--------------
Manages secure and resilient connection to InfluxDB cluster.
Supports token-based authentication, SSL, retries, and failover.
"""

from influxdb_client import InfluxDBClient
import yaml
from pathlib import Path
import logging
import time

logger = logging.getLogger("InfluxDBConnection")
logger.setLevel(logging.INFO)


class InfluxDBConnection:
    """Advanced connection handler for InfluxDB."""

    def __init__(self, config_path="config/databases/influxdb/config.yaml"):
        self.config = self._load_config(config_path)
        self.client = self._connect()

    def _load_config(self, path):
        """Load configuration YAML."""
        with open(Path(path), "r") as f:
            return yaml.safe_load(f)

    def _connect(self):
        """Connect to InfluxDB with retries & failover."""
        urls = [node["url"] for node in self.config["influxdb"]["nodes"]]
        token = self.config["influxdb"]["token"]
        org = self.config["influxdb"]["org"]

        retries, delay = 5, 2
        for attempt in range(retries):
            for url in urls:
                try:
                    client = InfluxDBClient(url=url, token=token, org=org, timeout=30_000)
                    health = client.health()
                    if health.status == "pass":
                        logger.info(f"✅ Connected to InfluxDB node {url}")
                        return client
                except Exception as e:
                    logger.warning(f"Connection attempt {attempt+1} to {url} failed: {e}")
                    time.sleep(delay)
            delay *= 2
        raise Exception("❌ Could not connect to any InfluxDB nodes")

    def get_client(self):
        return self.client
