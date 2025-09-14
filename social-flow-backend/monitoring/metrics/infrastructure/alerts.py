# Infrastructure alerts: thresholds, webhooks, escalation logic
"""
Alerting manager for infra metrics.

Provides:
 - InfraAlertManager which forwards severity messages to configured channels
 - Pluggable channel implementations for Slack, Email, PagerDuty
 - Rate-limiting / dedupe basic mechanism to avoid alert storms
"""

import logging
import time
from typing import List, Dict
from collections import deque
from .config import InfraMetricsConfig

logger = logging.getLogger("infra_metrics.alerts")


class InfraAlertManager:
    """
    Basic alert manager with deduping and rate limit window per alert_key.
    """

    def __init__(self, channels: List[str] = None, dedupe_window_seconds: int = 300):
        self.channels = channels or InfraMetricsConfig.ALERT_CHANNELS
        self.dedupe_window_seconds = dedupe_window_seconds
        # small in-memory dedupe cache: alert_key -> last_sent_ts
        self._last_sent: Dict[str, float] = {}
        # sliding list of recent alerts (for lightweight auditing)
        self._recent = deque(maxlen=1000)

    def should_send(self, key: str) -> bool:
        now = time.time()
        last = self._last_sent.get(key)
        if last and (now - last) < self.dedupe_window_seconds:
            logger.debug("Alert deduped for key=%s (last=%s)", key, last)
            return False
        self._last_sent[key] = now
        return True

    def trigger(self, key: str, message: str, severity: str = "warning"):
        """
        Trigger an alert if not deduped.
        key: unique key identifying the alert (e.g., infra_cpu_high_nodeA)
        """
        if not self.should_send(key):
            return
        self._recent.append((time.time(), key, severity, message))
        for ch in self.channels:
            try:
                if ch == "slack":
                    self._send_slack(message, severity)
                elif ch == "email":
                    self._send_email(message, severity)
                elif ch == "pagerduty":
                    self._send_pagerduty(message, severity)
                else:
                    logger.warning("Unknown alert channel configured: %s", ch)
            except Exception:
                logger.exception("Failed to send alert to %s", ch)

    def _send_slack(self, message: str, severity: str):
        # Placeholder — integrate with a Slack client or webhook
        logger.info("[ALERT][SLACK][%s] %s", severity.upper(), message)

    def _send_email(self, message: str, severity: str):
        # Placeholder — integrate with SMTP or transactional email provider
        logger.info("[ALERT][EMAIL][%s] %s", severity.upper(), message)

    def _send_pagerduty(self, message: str, severity: str):
        # Placeholder — integrate with PagerDuty Events API
        logger.info("[ALERT][PAGERDUTY][%s] %s", severity.upper(), message)
