# scripts/monitoring/metric_collector.py
import logging
import time
import psutil
import threading
from typing import Dict, Any, Callable

from .prometheus_exporter import PrometheusExporter
from .utils import retry

logger = logging.getLogger("monitoring.metrics")


class MetricCollector:
    """
    Host-level metrics collector (CPU, memory, disk, network). Pushes to Prometheus exporter.
    """

    def __init__(self, config: Dict[str, Any], exporter: PrometheusExporter):
        self.config = config
        self.exporter = exporter
        self.interval = int(config.get("monitoring", {}).get("metrics", {}).get("collection_interval", 15))
        # hold thread ref so runner can stop if needed
        self._thread = None
        self._stop = threading.Event()

        # register gauges
        self.exporter.register_gauge("host_cpu_percent", "Host CPU percent")
        self.exporter.register_gauge("host_mem_percent", "Host memory percent")
        self.exporter.register_gauge("host_disk_percent", "Host disk percent", labels=["mountpoint"])

    def collect_once(self):
        # cpu
        cpu = psutil.cpu_percent(interval=0.5)
        self.exporter.set_gauge("host_cpu_percent", cpu)
        logger.debug("Collected cpu: %s", cpu)

        # memory
        mem = psutil.virtual_memory().percent
        self.exporter.set_gauge("host_mem_percent", mem)
        logger.debug("Collected mem: %s", mem)

        # disk per partition (limited to root and /var)
        parts = [p.mountpoint for p in psutil.disk_partitions(all=False) if p.mountpoint in ("/", "/var", "/opt", "/home")]
        seen = set()
        for mount in parts:
            if mount in seen:
                continue
            seen.add(mount)
            try:
                usage = psutil.disk_usage(mount).percent
                self.exporter.set_gauge("host_disk_percent", usage, labels=[mount])
            except Exception:
                logger.exception("Failed to read disk usage for %s", mount)

    @retry(exception_types=(Exception,), max_attempts=3, backoff_factor=0.5)
    def _run_loop(self):
        logger.info("Starting metrics collection loop (interval=%s)", self.interval)
        while not self._stop.is_set():
            try:
                self.collect_once()
            except Exception:
                logger.exception("Error during metric collection")
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.info("Metric collector already running")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
