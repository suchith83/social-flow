import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from .config import report_settings
from .utils import logger


class EmailNotifier:
    """Send reports via email"""

    def send_report(self, to_email: str, subject: str, body: str, attachments: list[str] = None):
        msg = MIMEMultipart()
        msg["From"] = report_settings.SMTP_USER
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        if attachments:
            for file_path in attachments:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
                    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(file_path)}"'
                    msg.attach(part)

        with smtplib.SMTP(report_settings.SMTP_SERVER, report_settings.SMTP_PORT) as server:
            server.starttls()
            server.login(report_settings.SMTP_USER, report_settings.SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email report sent to {to_email}")
