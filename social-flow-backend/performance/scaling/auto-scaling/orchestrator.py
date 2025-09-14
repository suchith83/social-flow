# Core orchestrator for auto-scaling decisions
# performance/scaling/auto_scaling/orchestrator.py

import asyncio
import logging
from typing import Dict, Any

from .monitor import Monitor
from .executor import Executor
from .policies import ThresholdPolicy, PredictivePolicy
from .models import SimpleForecastModel
from .exceptions import AutoScalingError


logger = logging.getLogger("auto_scaling.orchestrator")


class Orchestrator:
    """
    Core orchestrator that coordinates monitoring, policy decision, and execution.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = Monitor()
        self.executor = Executor()

        policy_name = config["scaling"]["policy"]
        if policy_name == "threshold":
            self.policy = ThresholdPolicy()
        elif policy_name == "predictive":
            self.policy = PredictivePolicy(SimpleForecastModel())
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

    async def run_once(self, metrics: Dict[str, Any]):
        try:
            decision = self.policy.decide(
                metrics,
                self.executor.current_instances,
                self.config["scaling"]["min_instances"],
                self.config["scaling"]["max_instances"],
            )
            await self.executor.scale_to(decision)
        except Exception as e:
            raise AutoScalingError(f"Orchestration failed: {e}")

    async def start(self):
        """
        Start orchestrator loop.
        """
        async for metrics in self.monitor.stream_metrics(interval=5.0):
            await self.run_once(metrics)
