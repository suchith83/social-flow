# Regional health check engine
# performance/scaling/regional/healthcheck.py

import asyncio
import logging
from typing import Dict
from .exceptions import HealthCheckError


logger = logging.getLogger("regional.healthcheck")


class HealthChecker:
    """
    Periodically checks if regions are healthy.
    """

    def __init__(self, regions: Dict[str, Dict], interval: int = 15):
        self.regions = regions
        self.interval = interval

    async def check_region(self, region: str, data: Dict):
        try:
            await asyncio.sleep(0.05)
            # Fake health check: 90% healthy chance
            healthy = random.random() > 0.1
            self.regions[region]["healthy"] = healthy
            status = "healthy" if healthy else "unhealthy"
            logger.info(f"Region {region} health: {status}")
        except Exception as e:
            raise HealthCheckError(f"Failed health check {region}: {e}")

    async def run(self):
        while True:
            tasks = [self.check_region(r, d) for r, d in self.regions.items()]
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.interval)
