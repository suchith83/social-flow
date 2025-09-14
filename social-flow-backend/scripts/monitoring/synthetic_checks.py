# scripts/monitoring/synthetic_checks.py
import logging
import time
import threading
import requests
from typing import Dict, Any, List

from .prometheus_exporter import PrometheusExporter

logger = logging.getLogger("monitoring.synthetic")


class SyntheticChecker:
    """
    Run synthetic checks (HTTP GET/POST, simple DB pings) on a schedule and export metrics.
    Example config:
    monitoring:
      synthetic:
        interval: 60
        checks:
          - name: homepage
            type: http
            url: https://example.com/health
            method: GET
            expected_status: 200
            timeout: 5
    """

    def __init__(self, config: Dict[str, Any], exporter: PrometheusExporter, alert_cb=None):
        self.config = config
        self.exporter = exporter
        self.alert_cb = alert_cb
        self.interval = int(config.get("monitoring", {}).get("synthetic", {}).get("interval", 60))
        self.checks = config.get("monitoring", {}).get("synthetic", {}).get("checks", [])
        self._stop = threading.Event()
        self._thread = None

    def _run_http_check(self, check: Dict[str, Any]):
        name = check.get("name", "unnamed")
        url = check.get("url")
        method = (check.get("method") or "GET").upper()
        timeout = float(check.get("timeout", 5))
        expected = int(check.get("expected_status", 200))
        tries = int(check.get("retries", 1))

        start = time.time()
        try:
            resp = requests.request(method, url, timeout=timeout)
            latency = time.time() - start
            self.exporter.set_gauge("synthetic_check_latency_seconds", latency, labels=[name])
            if resp.status_code != expected:
                self.exporter.inc_counter("synthetic_check_failures_total", labels=[name])
                msg = f"Check {name} returned status {resp.status_code} expected {expected}"
                logger.warning(msg)
                if self.alert_cb:
                    self.alert_cb(msg)
            else:
                logger.debug("Check %s ok: status=%d latency=%.3fs", name, resp.status_code, latency)
        except Exception as e:
            latency = time.time() - start
            self.exporter.set_gauge("synthetic_check_latency_seconds", latency, labels=[name])
            self.exporter.inc_counter("synthetic_check_failures_total", labels=[name])
            msg = f"Check {name} failed: {e}"
            logger.exception(msg)
            if self.alert_cb:
                self.alert_cb(msg)

    def _run_check_once(self, check: Dict[str, Any]):
        t = check.get("type", "http").lower()
        if t == "http":
            self._run_http_check(check)
        else:
            logger.warning("Unsupported synthetic check type: %s", t)

    def _loop(self):
        logger.info("Starting synthetic checker loop (interval=%s)", self.interval)
        while not self._stop.is_set():
            for check in self.checks:
                try:
                    self._run_check_once(check)
                except Exception:
                    logger.exception("Error executing check: %s", check.get("name"))
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
