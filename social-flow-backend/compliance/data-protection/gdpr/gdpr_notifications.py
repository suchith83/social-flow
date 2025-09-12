"""
# User notifications regarding GDPR rights
"""
"""
GDPR Notification Service
-------------------------
"""

import smtplib
from email.mime.text import MIMEText
from .gdpr_exceptions import GDPRRequestError

class GDPRNotificationService:
    """Sends notifications to data subjects."""

    def __init__(self, smtp_server="localhost", smtp_port=25):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_notification(self, email: str, subject: str, message: str):
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = "dpo@company.com"
            msg["To"] = email

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())

            return {"status": "sent", "recipient": email}
        except Exception as e:
            raise GDPRRequestError(f"Failed to send GDPR notification: {e}")
