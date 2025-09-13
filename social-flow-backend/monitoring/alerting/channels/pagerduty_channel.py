# PagerDuty alerting channel
import requests
import logging
from .base_channel import BaseChannel

logger = logging.getLogger(__name__)

class PagerDutyChannel(BaseChannel):
    """
    PagerDuty alerting channel.
    Uses Events API v2 for triggering incidents.
    """

    def __init__(self, routing_key: str, severity: str = "critical", **kwargs):
        super().__init__(kwargs)
        self.routing_key = routing_key
        self.severity = severity

    def send_alert(self, message: str, source: str = "monitoring-system", **kwargs) -> bool:
        message = self.validate_message(message)
        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": message,
                "severity": self.severity,
                "source": source,
            },
        }
        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=5,
            )
            if response.status_code == 202:
                logger.info("PagerDuty alert sent successfully.")
                return True
            logger.error("Failed to send PagerDuty alert: %s", response.text)
            return False
        except requests.RequestException as e:
            logger.exception("Error sending PagerDuty alert: %s", e)
            return False
