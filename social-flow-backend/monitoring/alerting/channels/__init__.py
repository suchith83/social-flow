# Package initializer for channels module
"""
Monitoring Alerting Channels Package

This package provides multiple implementations of alerting channels such as Slack,
Email, SMS, Webhook, Teams, and PagerDuty. Each channel inherits from the
BaseChannel class, ensuring consistency in configuration and sending alerts.

Usage Example:
--------------
from monitoring.alerting.channels import SlackChannel, EmailChannel

slack = SlackChannel(webhook_url="https://hooks.slack.com/...")
slack.send_alert("🚨 Critical CPU Usage Detected!")

email = EmailChannel(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="alerts@example.com",
    password="super-secret",
    from_addr="alerts@example.com",
    to_addrs=["admin@example.com"]
)
email.send_alert("🚨 Database latency exceeds threshold!")
"""

from .base_channel import BaseChannel
from .slack_channel import SlackChannel
from .email_channel import EmailChannel
from .sms_channel import SMSChannel
from .webhook_channel import WebhookChannel
from .teams_channel import TeamsChannel
from .pagerduty_channel import PagerDutyChannel

__all__ = [
    "BaseChannel",
    "SlackChannel",
    "EmailChannel",
    "SMSChannel",
    "WebhookChannel",
    "TeamsChannel",
    "PagerDutyChannel",
]
