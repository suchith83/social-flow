from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from .config import settings
from .utils import logger


class Monitor:
    """Push metrics to Prometheus"""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.records_gauge = Gauge(
            "etl_records_processed", "Number of records processed", registry=self.registry
        )
        self.errors_gauge = Gauge(
            "etl_errors", "Number of errors encountered", registry=self.registry
        )

    def push_metrics(self, records_processed: int, errors: int = 0):
        self.records_gauge.set(records_processed)
        self.errors_gauge.set(errors)
        push_to_gateway(settings.PROMETHEUS_PUSHGATEWAY, job="etl-jobs", registry=self.registry)
        logger.info(f"Pushed metrics: {records_processed} records, {errors} errors")
