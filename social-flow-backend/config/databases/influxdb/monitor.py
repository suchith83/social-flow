"""Health checks, shard status, retention policy monitoring."""
"""
monitor.py
----------
Monitors cluster health, shards, and retention policies in InfluxDB.
"""

from .connection import InfluxDBConnection
import logging

logger = logging.getLogger("InfluxDBMonitor")
logger.setLevel(logging.INFO)


class InfluxDBMonitor:
    def __init__(self):
        self.client = InfluxDBConnection().get_client()

    def health_check(self):
        """Return cluster health status."""
        return self.client.health()

    def retention_policies(self):
        """List retention policies in the org."""
        return self.client.buckets_api().find_buckets().buckets

    def shard_status(self):
        """Simulate shard monitoring via query."""
        query = f'from(bucket:"socialflow-metrics") |> range(start: -1h) |> count()'
        return self.client.query_api().query(query, org="socialflow-org")
