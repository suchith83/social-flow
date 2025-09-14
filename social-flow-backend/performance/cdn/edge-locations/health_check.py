# Performs health checks for edge nodes
# performance/cdn/edge-locations/health_check.py
"""
Health checks for edge nodes.

Features:
- Asynchronous checks (HTTP/TCP/ICMP stub)
- Pluggable check functions
- Retry/backoff with exponential jitter
- Health state reporting to registry/monitoring integration
"""

from typing import Callable, Dict, Any, Optional
import asyncio
import aiohttp
from .utils import logger, sleep_backoff
from .registry import EdgeRegistry
import time

DEFAULT_TIMEOUT = 3

class HealthChecker:
    def __init__(self, registry: EdgeRegistry, concurrency: int = 50):
        self.registry = registry
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def http_check(self, node: Dict, path: str = "/health") -> bool:
        """Perform a simple HTTP GET health check on node hostname/IP."""
        url = f"http://{node['ip']}{path}"
        attempts = 0
        await self._ensure_session()
        while attempts < 3:
            attempts += 1
            try:
                async with self.session.get(url, timeout=DEFAULT_TIMEOUT) as resp:
                    if resp.status == 200:
                        logger.debug(f"HTTP health OK {node['id']} -> {url}")
                        return True
                    else:
                        logger.debug(f"HTTP health non-200 {resp.status} for {node['id']}")
            except Exception as e:
                logger.debug(f"HTTP health check failed for {node['id']}: {e}")
            await sleep_backoff(attempts)
        return False

    async def check_node(self, node: Dict, check_fn: Callable[[Dict], asyncio.Future]) -> bool:
        """Run check_fn under concurrency control and update registry accordingly."""
        async with self.semaphore:
            healthy = await check_fn(node)
            # update registry health flag
            try:
                await self.registry.update_load(node["id"], node.get("current_load_rps", 0.0))
                # fetch and set healthy flag (adapter-level record update)
                rec = await self.registry.adapter.get(node["id"])
                if rec:
                    rec["healthy"] = healthy
                    rec["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    await self.registry.adapter.upsert(node["id"], rec)
            except Exception as e:
                logger.error(f"Failed to update health for {node['id']}: {e}")
            return healthy

    async def run_periodic_checks(self, interval_seconds: int = 30, check_fn: Optional[Callable] = None):
        """Continuously run health checks across registry nodes."""
        check_fn = check_fn or self.http_check
        await self._ensure_session()
        while True:
            nodes = await self.registry.list_nodes(healthy_only=False)
            tasks = [self.check_node(n.to_dict(), check_fn) for n in nodes]
            # run concurrently but respecting semaphore limits
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Completed round of health checks, {len(nodes)} nodes inspected")
            await asyncio.sleep(interval_seconds)
