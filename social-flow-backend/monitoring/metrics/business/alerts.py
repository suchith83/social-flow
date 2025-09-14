# Defines alert rules for KPIs and notifies stakeholders
"""
Alerting for business metrics.
"""

import logging
from typing import List
from .config import BusinessMetricsConfig


class BusinessAlertManager:
    def __init__(self, channels: List[str] = None):
        self.channels = channels or BusinessMetricsConfig.ALERT_CHANNELS

    def trigger_alert(self, message: str, severity: str = "critical"):
        for channel in self.channels:
            if channel == "slack":
                self._send_slack(message, severity)
            elif channel == "email":
                self._send_email(message, severity)
            else:
                logging.warning(f"[BusinessAlertManager] Unsupported channel: {channel}")

    def _send_slack(self, message: str, severity: str):
        logging.info(f"[Business Slack] ({severity.upper()}): {message}")

    def _send_email(self, message: str, severity: str):
        logging.info(f"[Business Email] ({severity.upper()}): {message}")
