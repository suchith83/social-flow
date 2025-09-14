# Orchestrates monitoring + policy + dispatcher
# performance/scaling/regional/orchestrator.py

import asyncio
import logging
from typing import Dict, Any

from .monitor import Monitor
from .dispatcher import Dispatcher
from .policies import LatencyPolicy, WeightedPolicy, FailoverPolicy, GeoPolicy
from .healthcheck import HealthChecker
from .exceptions import RegionalError


logger = logging.getLogger("regional.orchestrator")


class Orchestrator:
    """
    Orchestrates multi-region routing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = Monitor(config["regions"])
        self.regions = self.monitor.regions
        self.dispatcher = Dispatcher(self.regions)

        policy_name = config["policy"]
        if policy_name == "latency":
            self.policy = LatencyPolicy()
        elif policy_name == "weighted":
            self.policy = WeightedPolicy()
        elif policy_name == "failover":
            self.policy = FailoverPolicy(list(config["regions"].keys()))
        elif policy_name == "geo":
            self.policy = GeoPolicy()
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        self.healthchecker = HealthChecker(self.regions, config["healthcheck_interval"])

    async def handle_request(self, request: Dict):
        try:
            await self.monitor.update_metrics()
            if isinstance(self.policy, GeoPolicy):
                region = self.policy.select(self.regions, request["client_ip"])
            else:
                region = self.policy.select(self.regions)
            await self.dispatcher.dispatch(region, request)
        except Exception as e:
            raise RegionalError(f"Request handling failed: {e}")

    async def start(self):
        asyncio.create_task(self.healthchecker.run())
        req_id = 0
        while True:
            req_id += 1
            request = {"id": req_id, "client_ip": f"10.0.0.{req_id % 255}"}
            await self.handle_request(request)
            await asyncio.sleep(1.0)
