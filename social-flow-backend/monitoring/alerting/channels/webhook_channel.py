# Generic webhook channel
import requests
import logging
from .base_channel import BaseChannel

logger = logging.getLogger(__name__)

class WebhookChannel(BaseChannel):
    """
    Generic webhook alerting channel.
    Sends POST requests with JSON payload.
    """

    def __init__(self, url: str, headers: dict | None = None, **kwargs):
        super().__init__(kwargs)
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send_alert(self, message: str, **kwargs) -> bool:
        message = self.validate_message(message)
        payload = {"alert": message, **kwargs}
        try:
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=5)
            if response.status_code in (200, 201, 202):
                logger.info("Webhook alert sent successfully.")
                return True
            logger.error("Webhook failed: %s", response.text)
            return False
        except requests.RequestException as e:
            logger.exception("Error sending webhook: %s", e)
            return False
