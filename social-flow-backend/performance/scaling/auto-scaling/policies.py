# Scaling policies (CPU, memory, custom metrics, schedules, predictive ML)
# performance/scaling/auto_scaling/policies.py

import logging
from typing import Dict, Any
from .exceptions import PolicyError


logger = logging.getLogger("auto_scaling.policies")


class ThresholdPolicy:
    """
    Threshold-based scaling policy.

    If CPU or memory exceeds threshold, scale up.
    If CPU or memory drops below threshold, scale down.
    """

    def __init__(self, up_threshold: float = 70.0, down_threshold: float = 30.0):
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def decide(self, metrics: Dict[str, Any], current_instances: int,
               min_instances: int, max_instances: int) -> int:
        try:
            cpu = metrics["cpu"]
            memory = metrics["memory"]

            if cpu > self.up_threshold or memory > self.up_threshold:
                return min(current_instances + 1, max_instances)
            elif cpu < self.down_threshold and memory < self.down_threshold:
                return max(current_instances - 1, min_instances)
            return current_instances
        except KeyError as e:
            raise PolicyError(f"Missing metric in decision: {e}")


class PredictivePolicy:
    """
    Predictive scaling policy using trained ML model.
    """

    def __init__(self, model):
        self.model = model

    def decide(self, metrics: Dict[str, Any], current_instances: int,
               min_instances: int, max_instances: int) -> int:
        try:
            forecast = self.model.forecast(metrics)
            logger.debug(f"Forecasted load: {forecast}")
            return min(max(forecast, min_instances), max_instances)
        except Exception as e:
            raise PolicyError(f"Predictive decision failed: {e}")
