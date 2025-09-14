# Tracks customer engagement metrics (sessions, churn, etc.)
"""
Tracks engagement KPIs like DAU/WAU/MAU and retention rates.
"""

from prometheus_client import Gauge
import threading


class EngagementMetrics:
    def __init__(self):
        self.dau = Gauge("business_dau", "Daily Active Users")
        self.wau = Gauge("business_wau", "Weekly Active Users")
        self.mau = Gauge("business_mau", "Monthly Active Users")
        self.retention_rate = Gauge("business_retention_rate", "User Retention Rate (0-1)")
        self._lock = threading.Lock()

    def update_dau(self, value: int):
        with self._lock:
            self.dau.set(value)

    def update_wau(self, value: int):
        with self._lock:
            self.wau.set(value)

    def update_mau(self, value: int):
        with self._lock:
            self.mau.set(value)

    def update_retention(self, rate: float):
        with self._lock:
            self.retention_rate.set(rate)
