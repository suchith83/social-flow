# Email integration channel
import smtplib
from email.mime.text import MIMEText
import logging
from .base_channel import BaseChannel

logger = logging.getLogger(__name__)

class EmailChannel(BaseChannel):
    """
    Email alerting channel via SMTP.
    """

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 from_addr: str, to_addrs: list[str], use_tls: bool = True, **kwargs):
        super().__init__(kwargs)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def send_alert(self, message: str, subject: str = "Monitoring Alert", **kwargs) -> bool:
        message = self.validate_message(message)
        mime_msg = MIMEText(message)
        mime_msg["Subject"] = subject
        mime_msg["From"] = self.from_addr
        mime_msg["To"] = ", ".join(self.to_addrs)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, mime_msg.as_string())
                logger.info("Email alert sent successfully.")
                return True
        except Exception as e:
            logger.exception("Failed to send email alert: %s", e)
            return False
