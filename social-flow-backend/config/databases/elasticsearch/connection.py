"""Secure, resilient Elasticsearch cluster connection with retry/failover."""
"""
connection.py
--------------
Manages secure connection to Elasticsearch cluster with retries,
SSL/TLS support, and node failover.
"""

from elasticsearch import Elasticsearch, exceptions
import yaml
from pathlib import Path
import logging
import time

logger = logging.getLogger("ElasticsearchConnection")
logger.setLevel(logging.INFO)


class ElasticsearchClient:
    """Advanced client wrapper for Elasticsearch."""

    def __init__(self, config_path="config/databases/elasticsearch/config.yaml"):
        self.config = self._load_config(config_path)
        self.client = self._connect()

    def _load_config(self, path):
        """Load YAML config."""
        with open(Path(path), "r") as f:
            return yaml.safe_load(f)

    def _connect(self):
        """Connect with retries and failover."""
        nodes = self.config["elasticsearch"]["nodes"]
        auth = (self.config["elasticsearch"]["user"], self.config["elasticsearch"]["password"])
        retries, delay = 5, 2
        for attempt in range(retries):
            try:
                client = Elasticsearch(
                    hosts=nodes,
                    basic_auth=auth,
                    verify_certs=True,
                    request_timeout=30,
                )
                if client.ping():
                    logger.info("✅ Connected to Elasticsearch cluster.")
                    return client
            except exceptions.ConnectionError as e:
                logger.warning(f"Connection failed (attempt {attempt+1}): {e}")
                time.sleep(delay)
                delay *= 2
        raise Exception("❌ Unable to connect to Elasticsearch cluster")

    def get_client(self):
        return self.client
