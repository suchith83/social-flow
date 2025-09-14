# Handles alert rules and notifications
"""
Alerting module that triggers notifications based on anomalies or thresholds.
Supports pluggable channels (email, Slack, PagerDuty, etc.).
"""

import logging
from typing import List
from .config import MetricsConfig


class AlertManager:
    """Handles alert triggering and routing."""

    def __init__(self, channels: List[str] = None):
        self.channels = channels or MetricsConfig.ALERT_CHANNELS

    def trigger_alert(self, message: str, severity: str = "warning"):
        """Trigger alert across configured channels."""
        for channel in self.channels:
            if channel == "slack":
                self._send_slack(message, severity)
            elif channel == "email":
                self._send_email(message, severity)
            else:
                logging.warning(f"[AlertManager] Unsupported channel: {channel}")

    def _send_slack(self, message: str, severity: str):
        logging.info(f"[Slack] ({severity.upper()}): {message}")

    def _send_email(self, message: str, severity: str):
        logging.info(f"[Email] ({severity.upper()}): {message}")
