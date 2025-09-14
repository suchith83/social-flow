# Health check engine (ping, HTTP, TCP)
# performance/scaling/load_balancing/healthcheck.py

import asyncio
import logging
import socket
from typing import Dict
from .exceptions import HealthCheckError


logger = logging.getLogger("load_balancing.healthcheck")


class HealthChecker:
    """
    Performs periodic health checks on backend nodes.
    """

    def __init__(self, nodes: Dict[str, Dict], interval: int = 10):
        self.nodes = nodes
        self.interval = interval

    async def _tcp_ping(self, host: str, port: int) -> bool:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def check_node(self, node_id: str, node: Dict):
        healthy = await self._tcp_ping(node["host"], node["port"])
        self.nodes[node_id]["healthy"] = healthy
        status = "healthy" if healthy else "unhealthy"
        logger.info(f"Health check {node_id}: {status}")

    async def run(self):
        while True:
            tasks = [self.check_node(node_id, node) for node_id, node in self.nodes.items()]
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.interval)
