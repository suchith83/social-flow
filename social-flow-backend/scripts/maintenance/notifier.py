# scripts/maintenance/notifier.py
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger("maintenance.notifier")

class Notifier:
    """
    Send simple notifications to Slack (webhook) or fallback to log.
    Extendable for email or other providers.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("maintenance", {}).get("notifications", {})
        self.slack_webhook = self.config.get("slack_webhook") or None
        self.enabled = bool(self.slack_webhook)

    def notify(self, message: str, level: str = "info"):
        logger.info("Notifier: %s", message)
        if not self.enabled:
            return
        payload = {"text": f"[maintenance] {message}"}
        try:
            r = requests.post(self.slack_webhook, json=payload, timeout=5)
            r.raise_for_status()
        except Exception as e:
            logger.exception("Failed to send notification: %s", e)
