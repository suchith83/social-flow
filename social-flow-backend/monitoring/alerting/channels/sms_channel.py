# SMS integration channel
import logging
from .base_channel import BaseChannel
from twilio.rest import Client

logger = logging.getLogger(__name__)

class SMSChannel(BaseChannel):
    """
    SMS alerting channel using Twilio API.
    """

    def __init__(self, account_sid: str, auth_token: str, from_number: str, to_numbers: list[str], **kwargs):
        super().__init__(kwargs)
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.to_numbers = to_numbers

    def send_alert(self, message: str, **kwargs) -> bool:
        message = self.validate_message(message)
        try:
            for number in self.to_numbers:
                self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=number
                )
            logger.info("SMS alerts sent successfully.")
            return True
        except Exception as e:
            logger.exception("Failed to send SMS alerts: %s", e)
            return False
