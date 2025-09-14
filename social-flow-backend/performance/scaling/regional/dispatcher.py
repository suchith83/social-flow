# Dispatches traffic to selected region
# performance/scaling/regional/dispatcher.py

import asyncio
import logging
from typing import Dict
from .exceptions import DispatchError


logger = logging.getLogger("regional.dispatcher")


class Dispatcher:
    """
    Dispatches traffic to selected region.
    """

    def __init__(self, regions: Dict[str, Dict]):
        self.regions = regions

    async def dispatch(self, region: str, request: Dict):
        if not self.regions[region]["healthy"]:
            raise DispatchError(f"Cannot dispatch to unhealthy region {region}")
        await asyncio.sleep(0.05)
        logger.info(f"Dispatched request {request['id']} to region {region} ({self.regions[region]['endpoint']})")
