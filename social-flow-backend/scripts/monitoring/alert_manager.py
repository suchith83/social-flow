# scripts/monitoring/alert_manager.py
import logging
import json
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger("monitoring.alerts")


class AlertManager:
    """
    Sends alerts to configured channels: Slack webhook, PagerDuty Events v2, or email (SMTP).
    Minimal implementation — extend in your environment with secure credential storage.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("monitoring", {}).get("alerts", {})
        self.slack_webhook = self.config.get("slack_webhook")
        self.pagerduty_key = self.config.get("pagerduty_key")
        self.email = self.config.get("email", {})  # dict containing smtp_host, from, to list

    def alert(self, subject: str, body: Optional[str] = None):
        body = body or subject
        logger.info("Alert: %s", subject)
        if self.slack_webhook:
            try:
                payload = {"text": f"[alert] {subject}\n{body}"}
                r = requests.post(self.slack_webhook, json=payload, timeout=5)
                r.raise_for_status()
            except Exception:
                logger.exception("Slack alert failed")

        if self.pagerduty_key:
            try:
                pd_payload = {
                    "routing_key": self.pagerduty_key,
                    "event_action": "trigger",
                    "payload": {
                        "summary": subject,
                        "source": "socialflow-monitor",
                        "severity": "critical",
                    },
                }
                r = requests.post("https://events.pagerduty.com/v2/enqueue", json=pd_payload, timeout=5)
                r.raise_for_status()
            except Exception:
                logger.exception("PagerDuty alert failed")

        # Email sending can be implemented via smtplib if required.
