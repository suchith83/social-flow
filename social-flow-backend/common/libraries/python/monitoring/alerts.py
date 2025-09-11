# alerts.py
import smtplib
import requests
from email.mime.text import MIMEText
from typing import Optional


class AlertManager:
    """
    Alerting system supporting:
    - Email (SMTP)
    - Slack (Webhook)
    - PagerDuty (Events API)
    """

    def __init__(self, slack_webhook: Optional[str] = None, smtp_server: Optional[str] = None,
                 smtp_user: Optional[str] = None, smtp_password: Optional[str] = None,
                 pagerduty_key: Optional[str] = None):
        self.slack_webhook = slack_webhook
        self.smtp_server = smtp_server
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.pagerduty_key = pagerduty_key

    def send_slack(self, message: str):
        if not self.slack_webhook:
            return
        requests.post(self.slack_webhook, json={"text": message})

    def send_email(self, subject: str, body: str, to: str):
        if not self.smtp_server or not self.smtp_user:
            return
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = to
        with smtplib.SMTP(self.smtp_server) as server:
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.smtp_user, [to], msg.as_string())

    def send_pagerduty(self, message: str, severity="critical"):
        if not self.pagerduty_key:
            return
        requests.post("https://events.pagerduty.com/v2/enqueue", json={
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": message,
                "severity": severity,
                "source": "monitoring-system"
            }
        })
