from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from .config import dashboard_settings
from .utils import logger


class DashboardMonitor:
    """Expose and push metrics for dashboards"""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.active_users = Gauge(
            "dashboard_active_users", "Number of active dashboard users", registry=self.registry
        )

    def push_metrics(self, active_user_count: int):
        self.active_users.set(active_user_count)
        push_to_gateway(
            dashboard_settings.PROMETHEUS_PUSHGATEWAY,
            job="predictive-dashboards",
            registry=self.registry,
        )
        logger.info(f"Pushed metrics: {active_user_count} active users")
