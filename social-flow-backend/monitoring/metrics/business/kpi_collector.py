# Collects business KPIs from different data sources
"""
Collector for core business KPIs like signups, churn, active users.
"""

from prometheus_client import Counter, Gauge
import threading


class KPICollector:
    """Thread-safe collector for high-level KPIs."""

    def __init__(self):
        self.signups = Counter("business_user_signups_total", "Total number of user signups")
        self.active_users = Gauge("business_active_users", "Number of active users")
        self.churned_users = Counter("business_user_churn_total", "Total number of churned users")
        self._lock = threading.Lock()

    def record_signup(self):
        with self._lock:
            self.signups.inc()

    def update_active_users(self, count: int):
        with self._lock:
            self.active_users.set(count)

    def record_churn(self):
        with self._lock:
            self.churned_users.inc()
