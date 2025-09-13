# Microsoft Teams integration channel
import requests
import logging
from .base_channel import BaseChannel

logger = logging.getLogger(__name__)

class TeamsChannel(BaseChannel):
    """
    Microsoft Teams alerting channel via Incoming Webhook.
    """

    def __init__(self, webhook_url: str, **kwargs):
        super().__init__(kwargs)
        self.webhook_url = webhook_url

    def send_alert(self, message: str, title: str = "Monitoring Alert", **kwargs) -> bool:
        message = self.validate_message(message)
        payload = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": title,
            "themeColor": "FF0000",
            "sections": [{"activityTitle": title, "text": message}],
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info("Teams alert sent successfully.")
                return True
            logger.error("Failed to send Teams alert: %s", response.text)
            return False
        except requests.RequestException as e:
            logger.exception("Error sending Teams alert: %s", e)
            return False
