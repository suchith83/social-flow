# Controls and optimizes traffic flow
# performance/cdn/optimization/traffic_shaping.py
"""
Traffic shaping and routing policies.
"""

import asyncio
import random
from typing import Dict, List
from .utils import logger

class TrafficShaper:
    def __init__(self, bandwidth_limit_mbps: float = 100.0):
        self.bandwidth_limit = bandwidth_limit_mbps
        self.current_usage = 0.0

    async def throttle(self, request_size_kb: float):
        """Throttle based on bandwidth availability."""
        mb_used = request_size_kb / 1024
        if self.current_usage + mb_used > self.bandwidth_limit:
            wait_time = mb_used / self.bandwidth_limit
            logger.debug(f"Throttling {request_size_kb:.1f}KB for {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        self.current_usage += mb_used
        await asyncio.sleep(0.001)  # simulate processing latency
        self.current_usage -= mb_used

    def route_request(self, edges: List[Dict]) -> Dict:
        """Simple weighted random edge selection."""
        if not edges:
            raise ValueError("No edges available")
        weights = [e.get("weight", 1) for e in edges]
        chosen = random.choices(edges, weights=weights, k=1)[0]
        logger.debug(f"Routed request to {chosen['id']}")
        return chosen
