# Trend analysis logic for executive metrics
"""
Trend Analysis

Analyzes KPIs for upward/downward trends and generates executive insights.
"""

import statistics
from typing import List, Optional


class TrendAnalyzer:
    def analyze(self, metric_name: str, values: List[float]) -> Optional[str]:
        if len(values) < 3:
            return None

        slope = values[-1] - values[0]
        mean = statistics.mean(values)

        if slope > 0.05 * mean:
            return "Positive trend (growth)"
        elif slope < -0.05 * mean:
            return "Negative trend (decline)"
        else:
            return "Stable trend"
