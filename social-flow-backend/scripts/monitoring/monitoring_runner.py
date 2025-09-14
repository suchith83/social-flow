# scripts/monitoring/monitoring_runner.py
import logging
import signal
import sys
from typing import Optional

from .config_loader import ConfigLoader
from .prometheus_exporter import PrometheusExporter
from .metric_collector import MetricCollector
from .log_monitor import LogMonitor
from .synthetic_checks import SyntheticChecker
from .alert_manager import AlertManager
from .dashboard_generator import DashboardGenerator

logger = logging.getLogger("monitoring.runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MonitoringRunner:
    """
    High-level orchestrator to start monitoring stack:
      - Prometheus exporter
      - Metric collector (host)
      - Log monitor (optional)
      - Synthetic checks (optional)
      - Dashboard generation
    """

    def __init__(self, config_path: str = "monitoring.yaml"):
        self.config = ConfigLoader(config_path).load()
        self.exporter = PrometheusExporter(self.config)
        self.alerts = AlertManager(self.config)
        self.metric_collector = MetricCollector(self.config, self.exporter)
        self.log_monitor = LogMonitor(self.config, alert_cb=self.alerts.alert)
        self.synthetic = SyntheticChecker(self.config, self.exporter, alert_cb=self.alerts.alert)
        self.dashboard_gen = DashboardGenerator(self.config)
        self._stopping = False

    def start(self):
        logger.info("Starting MonitoringRunner")
        # start exporter first
        self.exporter.start()
        # generate dashboard json
        try:
            self.dashboard_gen.generate()
        except Exception:
            logger.exception("Dashboard generation failed")

        # start components
        try:
            self.metric_collector.start()
            # Only start log monitor if paths are configured
            if self.config.get("monitoring", {}).get("logs", {}).get("paths"):
                self.log_monitor.start()
            # Start synthetic checks only if configured
            if self.config.get("monitoring", {}).get("synthetic", {}).get("checks"):
                self.synthetic.start()
            logger.info("MonitoringRunner started successfully")
        except Exception:
            logger.exception("Failed to start monitoring components")
            self.stop()
            raise

        # graceful shutdown handling
        def _handle(signum, frame):
            logger.info("Signal %s received: stopping", signum)
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)

    def stop(self):
        if self._stopping:
            return
        self._stopping = True
        logger.info("Stopping MonitoringRunner")
        try:
            self.synthetic.stop()
        except Exception:
            logger.exception("Error stopping synthetic")
        try:
            self.log_monitor.stop()
        except Exception:
            logger.exception("Error stopping log monitor")
        try:
            self.metric_collector.stop()
        except Exception:
            logger.exception("Error stopping metric collector")
        logger.info("All monitoring components stopped.")


def main(argv=None):
    runner = MonitoringRunner()
    runner.start()
    # keep the main thread alive while background threads run
    try:
        while True:
            signal.pause()
    except Exception:
        runner.stop()


if __name__ == "__main__":
    main()
