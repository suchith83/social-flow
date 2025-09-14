# Orchestrates monitoring + algorithms + dispatcher
# performance/scaling/load_balancing/orchestrator.py

import asyncio
import logging
from typing import Dict, Any

from .monitor import Monitor
from .dispatcher import Dispatcher
from .algorithms import RoundRobin, LeastConnections, WeightedRoundRobin, ConsistentHashing
from .session_affinity import StickySession
from .healthcheck import HealthChecker
from .exceptions import LoadBalancingError


logger = logging.getLogger("load_balancing.orchestrator")


class Orchestrator:
    """
    Orchestrates monitoring, algorithm selection, health checks, and dispatch.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = Monitor(config["balancer"]["nodes"])
        self.nodes = self.monitor.nodes
        self.dispatcher = Dispatcher(self.nodes)

        algo = config["balancer"]["algorithm"]
        if algo == "round_robin":
            self.algo = RoundRobin(self.nodes)
        elif algo == "least_connections":
            self.algo = LeastConnections(self.nodes)
        elif algo == "weighted_round_robin":
            self.algo = WeightedRoundRobin(self.nodes)
        elif algo == "consistent_hashing":
            self.algo = ConsistentHashing(self.nodes)
        elif algo == "sticky":
            self.algo = StickySession(self.nodes)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        self.healthchecker = HealthChecker(self.nodes, config["balancer"]["healthcheck_interval"])

    async def handle_request(self, request: Dict):
        try:
            if isinstance(self.algo, (ConsistentHashing, StickySession)):
                node_id = self.algo.select(request.get("client_ip", "0.0.0.0"))
            else:
                node_id = self.algo.select()
            await self.dispatcher.dispatch(node_id, request)
        except Exception as e:
            raise LoadBalancingError(f"Request handling failed: {e}")

    async def start(self):
        asyncio.create_task(self.healthchecker.run())
        request_id = 0
        while True:
            request_id += 1
            request = {"id": request_id, "client_ip": f"192.168.0.{request_id % 5}"}
            await self.handle_request(request)
            await asyncio.sleep(0.5)
