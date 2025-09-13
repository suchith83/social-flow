# Adapter to collect and normalize KPI metrics
"""
KPI Adapter

Fetches high-level KPIs such as revenue, active users, conversion rates,
and uptime. Can integrate with BI tools, data warehouses, or APIs.
"""

import random
import time
from typing import List, Dict


class KPIAdapter:
    def __init__(self):
        self.cache: Dict[str, List[float]] = {}

    def fetch_kpi(self, metric_name: str, points: int = 7) -> List[float]:
        """
        Fetches KPI values. In production, connect to BI/ETL systems.
        """
        now = int(time.time())
        key = f"{metric_name}:{now // 60}"

        if key in self.cache:
            return self.cache[key]

        # Simulated business data
        if "revenue" in metric_name:
            values = [round(random.uniform(8000, 20000), 2) for _ in range(points)]
        elif "users_active" in metric_name:
            values = [random.randint(1000, 5000) for _ in range(points)]
        elif "conversion_rate" in metric_name:
            values = [round(random.uniform(0.5, 3.0), 2) for _ in range(points)]
        elif "uptime" in metric_name:
            values = [round(random.uniform(97.5, 100.0), 2) for _ in range(points)]
        else:
            values = [round(random.uniform(0, 1000), 2) for _ in range(points)]

        self.cache[key] = values
        return values
