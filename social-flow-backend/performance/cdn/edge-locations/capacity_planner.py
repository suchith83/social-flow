# Plans and allocates capacity across edge sites
# performance/cdn/edge-locations/capacity_planner.py
"""
Capacity planner

- Estimates capacity need per region / POP based on historical demand
- Suggests scaling actions (add edge, increase instance size, shed non-critical traffic)
- Uses exponential smoothing for demand forecasting
- Exposes an API to evaluate current headroom vs predicted demand
"""

from typing import List, Dict, Tuple
import math
from .utils import logger

def exponential_smoothing(series: List[float], alpha: float = 0.3) -> List[float]:
    """Simple single-exponential smoothing (returns smoothed series)."""
    if not series:
        return []
    s = [series[0]]
    for t in range(1, len(series)):
        s.append(alpha * series[t] + (1 - alpha) * s[-1])
    return s

def forecast_next(series: List[float], alpha: float = 0.3) -> float:
    """Forecast next value using exponential smoothing."""
    s = exponential_smoothing(series, alpha=alpha)
    return s[-1] if s else 0.0

def estimate_needed_capacity(current_capacity_rps: float, historical_rps: List[float], safety_margin: float = 1.25) -> Dict[str, float]:
    """
    Return capacity suggestions:
      - predicted_rps: forecasted next-second RPS
      - required_capacity: predicted * safety_margin
      - scale_up_by: max(0, required - current_capacity)
    """
    predicted = forecast_next(historical_rps)
    required = predicted * safety_margin
    scale_up = max(0.0, required - current_capacity_rps)
    logger.info(f"Predicted RPS {predicted:.2f}, required {required:.2f}, current {current_capacity_rps:.2f}")
    return {
        "predicted_rps": predicted,
        "required_capacity_rps": required,
        "scale_up_by_rps": scale_up
    }

def recommend_actions(current_capacity_rps: float, historical_rps: List[float], node_templates: List[Dict]) -> List[Dict]:
    """
    Given available node templates (each with capacity_rps and cost), recommend a minimal-cost set of templates
    to satisfy predicted demand. This is a knapsack-ish approximate solver (greedy by capacity/cost ratio).
    """
    needed = estimate_needed_capacity(current_capacity_rps, historical_rps)["scale_up_by_rps"]
    if needed <= 0:
        return []
    # compute value = capacity / cost
    choices = sorted(node_templates, key=lambda x: - (x.get("capacity_rps", 0) / (x.get("cost", 1.0))))
    plan = []
    accumulated = 0.0
    for t in choices:
        if accumulated >= needed:
            break
        plan.append(t)
        accumulated += t.get("capacity_rps", 0)
    logger.info(f"Recommended {len(plan)} templates to scale by ~{accumulated:.2f} rps")
    return plan
