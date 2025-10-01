"""Cluster health, node stats, shard allocation monitoring."""
"""
monitor.py
----------
Cluster health, node status, and shard allocation monitoring.
"""

from .connection import ElasticsearchClient
import logging

logger = logging.getLogger("ElasticsearchMonitor")
logger.setLevel(logging.INFO)


class ElasticsearchMonitor:
    def __init__(self):
        self.client = ElasticsearchClient().get_client()

    def get_cluster_health(self):
        """Check cluster health."""
        return self.client.cluster.health()

    def get_node_stats(self):
        """Get stats for all nodes."""
        return self.client.nodes.stats()

    def get_shard_allocation(self):
        """Get shard allocation across nodes."""
        return self.client.cat.shards(format="json")
