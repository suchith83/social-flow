# Simulation with synthetic requests
# performance/scaling/sharding/simulator.py

import asyncio
import logging
from .config import Config
from .orchestrator import Orchestrator


logger = logging.getLogger("sharding.simulator")


class Simulator:
    """
    Simulator for testing sharding strategies with synthetic requests.
    """

    def __init__(self, config_path: str):
        self.config = Config.load(config_path)
        self.orchestrator = Orchestrator(self.config)

    async def run(self):
        await self.orchestrator.start()
