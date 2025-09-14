# Simulation framework to test scaling strategies safely
# performance/scaling/auto_scaling/simulator.py

import asyncio
import logging
import random
from .orchestrator import Orchestrator
from .config import Config


logger = logging.getLogger("auto_scaling.simulator")


class Simulator:
    """
    Simulator for testing scaling strategies with synthetic workloads.
    """

    def __init__(self, config_path: str):
        self.config = Config.load(config_path)
        self.orchestrator = Orchestrator(self.config)

    async def synthetic_workload(self):
        """
        Generate synthetic workload by altering metrics.
        """
        while True:
            metrics = {
                "cpu": random.uniform(10, 90),
                "memory": random.uniform(20, 80),
                "network": random.randint(10**6, 10**7)
            }
            await self.orchestrator.run_once(metrics)
            await asyncio.sleep(3)

    async def run(self):
        """
        Run simulation with synthetic workload.
        """
        await self.synthetic_workload()
