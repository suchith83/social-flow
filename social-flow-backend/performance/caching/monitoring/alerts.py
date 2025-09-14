import logging
from typing import Dict

logger = logging.getLogger(__name__)

class CacheAlertManager:
    """
    Handles alert generation when anomalies or thresholds are breached.
    """

    def __init__(self, alert_channels=None):
        self.alert_channels = alert_channels or []

    def add_channel(self, channel):
        """Add an alerting channel (e.g., Slack, Email)."""
        self.alert_channels.append(channel)

    def trigger_alert(self, message: str):
        """Send alert to all channels."""
        logger.warning(f"ALERT: {message}")
        for channel in self.alert_channels:
            try:
                channel.send(message)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")


class SlackChannel:
    """Sends alerts to Slack via webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str):
        import requests
        response = requests.post(
            self.webhook_url,
            json={"text": message},
            timeout=5,
        )
        response.raise_for_status()


class EmailChannel:
    """Sends alerts via email (using SMTP)."""

    def __init__(self, smtp_server: str, from_addr: str, to_addr: str):
        self.smtp_server = smtp_server
        self.from_addr = from_addr
        self.to_addr = to_addr

    def send(self, message: str):
        import smtplib
        with smtplib.SMTP(self.smtp_server) as server:
            email_msg = f"Subject: Cache Alert\n\n{message}"
            server.sendmail(self.from_addr, self.to_addr, email_msg)
