# Predictive scaling models (time-series, regression, anomaly detection)
# performance/scaling/auto_scaling/models.py

import logging
import random
from typing import Dict


logger = logging.getLogger("auto_scaling.models")


class SimpleForecastModel:
    """
    Simple heuristic forecasting model.
    """

    def forecast(self, metrics: Dict[str, float]) -> int:
        # A fake forecast: use cpu load as proxy for desired instance count
        cpu_load = metrics.get("cpu", 0.0)
        base = int(cpu_load / 10)
        jitter = random.randint(-1, 1)
        prediction = max(1, base + jitter)
        logger.debug(f"Forecast based on CPU={cpu_load}: {prediction}")
        return prediction
