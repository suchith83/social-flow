# Handles alert creation, formatting, and notification routing
"""
Alert manager for security incidents.

- Dedupe and rate-limit to avoid alert storms
- Map severity levels to channels automatically
- Provide a simple escalation hook for on-call systems (PagerDuty)
"""

import logging
import time
from typing import List, Dict
from collections import deque
from .config import SecurityMetricsConfig
from .utils import hash_event_id

logger = logging.getLogger("security_metrics.alerts")

class SecurityAlertManager:
    def __init__(self, channels: List[str] = None, dedupe_window_seconds: int = None):
        self.channels = channels or SecurityMetricsConfig.ALERT_CHANNELS
        self.dedupe_window_seconds = dedupe_window_seconds or SecurityMetricsConfig.ALERT_DEDUPE_WINDOW_SECONDS
        self._last_sent: Dict[str, float] = {}
        self._recent = deque(maxlen=2000)

    def _should_send(self, key: str) -> bool:
        now = time.time()
        last = self._last_sent.get(key)
        if last and (now - last) < self.dedupe_window_seconds:
            logger.debug("Alert deduped: %s", key)
            return False
        self._last_sent[key] = now
        return True

    def trigger(self, title: str, body: str, severity: str = "high", dedupe_key: str = None):
        """
        Trigger an alert. For security, severity default is 'high'.
        dedupe_key: optional custom string to dedupe similar incidents (e.g., ip:1.2.3.4)
        """
        key = dedupe_key or hash_event_id(title, body)
        if not self._should_send(key):
            return

        self._recent.append((time.time(), key, severity, title))
        # route based on severity
        for ch in self.channels:
            try:
                if ch == "pagerduty":
                    self._send_pagerduty(title, body, severity)
                elif ch == "slack":
                    self._send_slack(title, body, severity)
                elif ch == "email":
                    self._send_email(title, body, severity)
                else:
                    logger.warning("Unknown alert channel: %s", ch)
            except Exception:
                logger.exception("Failed to send alert to %s", ch)

    def _send_pagerduty(self, title: str, body: str, severity: str):
        # Placeholder integration with PagerDuty Events API v2
        logger.info("[PAGERDUTY][%s] %s - %s", severity.upper(), title, body)

    def _send_slack(self, title: str, body: str, severity: str):
        # Placeholder slack webhook / client integration
        logger.info("[SLACK][%s] %s - %s", severity.upper(), title, body)

    def _send_email(self, title: str, body: str, severity: str):
        # Placeholder email send (SMTP or provider)
        logger.info("[EMAIL][%s] %s - %s", severity.upper(), title, body)
