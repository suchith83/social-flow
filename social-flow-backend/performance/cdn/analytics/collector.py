# Collects raw CDN logs, metrics, traces
# performance/cdn/analytics/collector.py
"""
Collector module
================
Collects CDN logs, telemetry, and metrics from multiple sources
such as edge servers, APIs, and streaming pipelines.
"""

import asyncio
import aiohttp
import random
from typing import AsyncGenerator, Dict
from .utils import logger, now_utc

class CDNCollector:
    def __init__(self, endpoints: list[str], batch_size: int = 100):
        self.endpoints = endpoints
        self.batch_size = batch_size

    async def fetch_endpoint(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Fetch logs/metrics from a CDN endpoint asynchronously."""
        try:
            async with session.get(url, timeout=5) as resp:
                data = await resp.json()
                logger.debug(f"Fetched data from {url}")
                return {"endpoint": url, "timestamp": now_utc(), "data": data}
        except Exception as e:
            logger.error(f"Failed to fetch from {url}: {e}")
            return {"endpoint": url, "timestamp": now_utc(), "error": str(e)}

    async def collect(self) -> AsyncGenerator[Dict, None]:
        """Continuously collect logs from all endpoints in batches."""
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [self.fetch_endpoint(session, url) for url in self.endpoints]
                results = await asyncio.gather(*tasks)
                for r in results:
                    yield r
                await asyncio.sleep(random.uniform(1, 3))  # jittered interval
