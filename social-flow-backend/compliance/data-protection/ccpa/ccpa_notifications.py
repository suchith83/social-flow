"""
# Send notifications about user rights
"""
"""
CCPA Notification Service
-------------------------
Handles sending notifications to consumers regarding their 
privacy rights and request statuses.
"""

import smtplib
from email.mime.text import MIMEText
from .ccpa_exceptions import CCPANotificationError

class CCPANotificationService:
    """Sends notifications related to CCPA rights."""

    def __init__(self, smtp_server="localhost", smtp_port=25):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_notification(self, email: str, subject: str, message: str):
        """
        Send an email notification.
        """
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = "privacy@company.com"
            msg["To"] = email

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.sendmail(msg["From"], [msg["To"]], msg.as_string())

            return {"status": "sent", "recipient": email}
        except Exception as e:
            raise CCPANotificationError(f"Failed to send notification: {e}")
