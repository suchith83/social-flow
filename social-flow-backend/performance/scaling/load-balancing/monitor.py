# Collects backend node health and load metrics
# performance/scaling/load_balancing/monitor.py

import asyncio
import logging
from typing import List, Dict


logger = logging.getLogger("load_balancing.monitor")


class Monitor:
    """
    Collects real-time node load metrics.
    In production this could integrate with Prometheus, custom APIs, or agent-based metrics.
    """

    def __init__(self, nodes: List[Dict]):
        self.nodes = {f"{n['host']}:{n['port']}": {"load": 0, "healthy": True, **n} for n in nodes}

    async def update_metrics(self, node_id: str, load: float):
        """
        Update node load manually (from dispatcher feedback).
        """
        if node_id in self.nodes:
            self.nodes[node_id]["load"] = load
            logger.debug(f"Updated {node_id} load={load}")

    async def get_nodes(self) -> Dict[str, Dict]:
        """
        Returns snapshot of nodes.
        """
        return self.nodes
