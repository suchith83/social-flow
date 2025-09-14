# Computes funnel progression metrics (conversion rates, drop-offs)
"""
Tracks conversion funnel metrics (CTR, checkout success, drop-off).
"""

from prometheus_client import Gauge
import threading


class FunnelMetrics:
    def __init__(self):
        self.ctr = Gauge("business_ctr", "Click-through rate (0-1)")
        self.conversion_rate = Gauge("business_conversion_rate", "Conversion rate (0-1)")
        self.checkout_success_rate = Gauge("business_checkout_success_rate", "Checkout success rate (0-1)")
        self._lock = threading.Lock()

    def update_ctr(self, value: float):
        with self._lock:
            self.ctr.set(value)

    def update_conversion_rate(self, value: float):
        with self._lock:
            self.conversion_rate.set(value)

    def update_checkout_success(self, value: float):
        with self._lock:
            self.checkout_success_rate.set(value)
