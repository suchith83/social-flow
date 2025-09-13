# Plan capacity based on trends and forecasts
"""
Capacity Planner

- Uses a small forecasting technique (linear regression on recent samples) to estimate short-term capacity needs.
- Returns actionable recommendations (scale-up, scale-down, no-change) with suggested magnitudes.
- Note: For production, integrate with real demand forecasting libraries (Prophet, ARIMA, LSTMs).
"""

from typing import Dict, List, Any
import statistics


class CapacityPlanner:
    def __init__(self, headroom_target: float = 0.2):
        """
        headroom_target: desired headroom fraction (e.g., 0.2 => 20% spare capacity)
        """
        self.headroom = headroom_target

    def estimate_capacity_requirements(self, data_bundle: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        """
        For each resource-like metric, estimate next-step requirement.
        Returns a dict: resource_key -> { recommendation, current, forecast, suggested_extra_units }
        """
        report = {}
        for metric, values in data_bundle.items():
            if not values or len(values) < 3:
                continue
            # choose heuristics only for known resource metrics
            if any(tok in metric.lower() for tok in ("cpu", "memory", "disk", "iops", "throughput", "requests")):
                current = values[-1]
                forecast = self._linear_forecast(values)
                recommendation, magnitude = self._decide_recommendation(current, forecast)
                report[metric] = {
                    "current": round(current, 2),
                    "forecast": round(forecast, 2),
                    "recommendation": recommendation,
                    "suggested_scale_pct": round(magnitude * 100, 2)
                }
        return report

    def _linear_forecast(self, values: List[float], steps: int = 1) -> float:
        """
        Apply a tiny linear regression (slope over index) to estimate next-step.
        It is intentionally simple: slope = (last - first)/N -> forecast = last + slope*steps
        """
        n = len(values) - 1
        if n <= 0:
            return values[-1]
        slope = (values[-1] - values[0]) / n
        return values[-1] + slope * steps

    def _decide_recommendation(self, current: float, forecast: float) -> (str, float):
        """
        Decide if we should scale. Magnitude returned is fraction (e.g., 0.2 => 20%).
        """
        # If forecasted usage exceeds (1 - headroom) * capacity (we interpret current as percentage)
        threshold = (1.0 - self.headroom) * 100.0
        # If metrics are not percentages, we still use relative logic: if forecast > current * 1.1 then scale
        if 0 <= current <= 100 and 0 <= forecast <= 100:
            if forecast >= threshold:
                # scale up to restore headroom
                suggested = (forecast / (1.0 - self.headroom)) - 1.0
                return "scale_up", max(0.05, suggested)
            elif forecast < threshold * 0.6:
                # safe to scale down slightly
                return "scale_down", 0.05
            else:
                return "no_change", 0.0
        else:
            # non-percent metrics: relative change threshold
            if forecast > current * 1.15:
                return "scale_up", min(1.0, (forecast / current) - 1.0)
            elif forecast < current * 0.85:
                return "scale_down", min(0.5, 1.0 - (forecast / current))
            else:
                return "no_change", 0.0
