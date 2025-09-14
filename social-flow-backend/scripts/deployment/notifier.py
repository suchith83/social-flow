# scripts/deployment/notifier.py
import logging
import requests
from typing import Dict, Any


class Notifier:
    """
    Sends notifications to Slack, Teams, or Email about deployment status.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook = config.get("notifications", {}).get("slack_webhook")

    def notify(self, message: str):
        logging.info(f"🔔 Notification: {message}")
        if self.webhook:
            try:
                requests.post(self.webhook, json={"text": message}, timeout=5)
            except Exception as e:
                logging.error(f"Failed to send Slack notification: {e}")
