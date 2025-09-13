# Slack integration channel
import requests
import logging
from .base_channel import BaseChannel

logger = logging.getLogger(__name__)

class SlackChannel(BaseChannel):
    """
    Slack alerting channel using Incoming Webhooks.
    """

    def __init__(self, webhook_url: str, **kwargs):
        super().__init__(kwargs)
        self.webhook_url = webhook_url

    def send_alert(self, message: str, **kwargs) -> bool:
        message = self.validate_message(message)
        payload = {"text": message, **kwargs}
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info("Slack alert sent successfully.")
                return True
            logger.error("Failed to send Slack alert: %s", response.text)
            return False
        except requests.RequestException as e:
            logger.exception("Error sending Slack alert: %s", e)
            return False
