# Simulation with synthetic workloads
# performance/scaling/regional/simulator.py

import asyncio
import logging
from .config import Config
from .orchestrator import Orchestrator


logger = logging.getLogger("regional.simulator")


class Simulator:
    """
    Simulator for testing regional routing with synthetic requests.
    """

    def __init__(self, config_path: str):
        self.config = Config.load(config_path)
        self.orchestrator = Orchestrator(self.config)

    async def run(self):
        await self.orchestrator.start()
