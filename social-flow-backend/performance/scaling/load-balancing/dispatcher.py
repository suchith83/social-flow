# Dispatches requests to selected nodes
# performance/scaling/load_balancing/dispatcher.py

import asyncio
import logging
from typing import Dict
from .exceptions import DispatchError


logger = logging.getLogger("load_balancing.dispatcher")


class Dispatcher:
    """
    Dispatches requests to selected backend nodes.
    """

    def __init__(self, nodes: Dict[str, Dict]):
        self.nodes = nodes

    async def dispatch(self, node_id: str, request: Dict):
        """
        Simulate dispatching a request to a node.
        """
        try:
            if not self.nodes[node_id]["healthy"]:
                raise DispatchError(f"Node {node_id} is unhealthy")
            await asyncio.sleep(0.05)  # simulate network
            logger.info(f"Dispatched request {request['id']} to {node_id}")
            self.nodes[node_id]["load"] += 1
            await asyncio.sleep(0.2)
            self.nodes[node_id]["load"] -= 1
        except Exception as e:
            raise DispatchError(f"Failed to dispatch: {e}")
