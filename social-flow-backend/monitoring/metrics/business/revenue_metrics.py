# Calculates and tracks revenue-related metrics
"""
Tracks revenue-related metrics (MRR, ARPU, LTV, transactions).
"""

from prometheus_client import Counter, Gauge
import threading


class RevenueMetrics:
    def __init__(self):
        self.revenue = Counter("business_revenue_total", "Total revenue collected", ["currency"])
        self.transactions = Counter("business_transactions_total", "Total transactions", ["currency"])
        self.mrr = Gauge("business_mrr", "Monthly Recurring Revenue")
        self.arpu = Gauge("business_arpu", "Average Revenue Per User")
        self.ltv = Gauge("business_ltv", "Customer Lifetime Value")
        self._lock = threading.Lock()

    def record_transaction(self, amount: float, currency: str = "USD"):
        with self._lock:
            self.revenue.labels(currency).inc(amount)
            self.transactions.labels(currency).inc()

    def update_mrr(self, value: float):
        with self._lock:
            self.mrr.set(value)

    def update_arpu(self, value: float):
        with self._lock:
            self.arpu.set(value)

    def update_ltv(self, value: float):
        with self._lock:
            self.ltv.set(value)
