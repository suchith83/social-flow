# Round-robin escalation implementation
"""
Round-robin and sequential escalators.

This file provides a RoundRobinEscalator that cycles targets inside a level
to distribute notifications across multiple users, and a SequentialEscalator
(example) that notifies all configured targets in order.

Important notes:
- These implementations are synchronous and blocking for simplicity.
- For production, wire this to a task queue (Celery/RQ/Kafka) or async event loop.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import time
import logging

from .escalator import Escalator
from .escalation_policy import EscalationLevel, EscalationPolicy
from .incident import Incident, IncidentStatus
from .persistence import PersistenceAdapter
from .retry_backoff import RetryBackoff

# We import channel implementations dynamically to remain flexible
from ..channels import SlackChannel, EmailChannel, SMSChannel, WebhookChannel, TeamsChannel, PagerDutyChannel  # type: ignore

logger = logging.getLogger(__name__)


class RoundRobinEscalator(Escalator):
    """
    Round-robin escalator cycles through targets inside each level.
    It will attempt to notify targets in round-robin order and escalate if not acknowledged.

    The `target` format should be a dict that indicates channel type and config, e.g.:
      {"type": "slack", "webhook": "https://...", "fallback": {...}}
    """

    def __init__(self, policy: EscalationPolicy, persistence: PersistenceAdapter = None, retry_backoff: Optional[RetryBackoff] = None):
        super().__init__(policy, persistence=persistence, retry_backoff=retry_backoff)
        # per-policy cursor to remember round-robin positions
        self._cursors: Dict[str, int] = {}

    def _build_channel(self, target: Dict[str, Any]) -> Optional[object]:
        """
        Factory to build a channel instance from target descriptor.
        Returns an object with a .send_alert(message, **kwargs) method.
        """
        ttype = target.get("type", "").lower()
        try:
            if ttype == "slack":
                return SlackChannel(webhook_url=target["webhook"])
            if ttype == "email":
                return EmailChannel(
                    smtp_host=target["smtp_host"],
                    smtp_port=target.get("smtp_port", 587),
                    username=target.get("username"),
                    password=target.get("password"),
                    from_addr=target.get("from_addr"),
                    to_addrs=target.get("to_addrs", []),
                    use_tls=target.get("use_tls", True),
                )
            if ttype == "sms":
                return SMSChannel(
                    account_sid=target["account_sid"],
                    auth_token=target["auth_token"],
                    from_number=target["from_number"],
                    to_numbers=target.get("to_numbers", []),
                )
            if ttype == "webhook":
                return WebhookChannel(url=target["url"], headers=target.get("headers"))
            if ttype == "teams":
                return TeamsChannel(webhook_url=target["webhook"])
            if ttype == "pagerduty":
                return PagerDutyChannel(routing_key=target["routing_key"], severity=target.get("severity", "critical"))
        except Exception as e:
            logger.exception("Error building channel for target %s: %s", target, e)
            return None
        logger.warning("Unsupported target type: %s", ttype)
        return None

    def _notify_target(self, channel_obj, message: str, max_retries: int = 1, backoff: Optional[RetryBackoff] = None) -> bool:
        """
        Attempts to send notification via the channel object with retries/backoff.
        Returns True if successful.
        """
        backoff = backoff or self.retry_backoff
        attempt = 1
        while attempt <= max_retries:
            try:
                # Channels implement .send_alert -> bool
                ok = channel_obj.send_alert(message)
                if ok:
                    logger.info("Notification succeeded on attempt %d", attempt)
                    return True
                else:
                    logger.warning("Notification returned False on attempt %d", attempt)
            except Exception as e:
                logger.exception("Notification attempt %d failed: %s", attempt, e)
            # compute sleep before next attempt
            sleep_for = backoff.next_backoff(attempt)
            logger.debug("Sleeping %s seconds before retrying", sleep_for)
            time.sleep(sleep_for)
            attempt += 1
        logger.error("All %d attempts failed for channel %s", max_retries, getattr(channel_obj, "__class__", None))
        return False

    def escalate(self, incident: Incident, stop_on_ack: bool = True, **kwargs) -> None:
        """
        Apply the escalation policy to the incident.

        Behavior:
          - For each level in the policy in order:
              * pick a target using round-robin
              * try to notify it (with level.retries and level.retry_backoff)
              * persist audit logs and update incident history
              * if acknowledged (simulated by incident.status change externally) stop if configured
              * wait level.timeout then move to next level
        """
        self.persistence.save_incident(incident)
        logger.info("Starting escalation for incident %s using policy %s", incident.id, self.policy.name)

        for idx, level in enumerate(self.policy.levels):
            if incident.status != IncidentStatus.OPEN:
                logger.info("Incident %s status is %s — stopping escalation", incident.id, incident.status)
                return

            logger.info("Escalating incident %s to level %s (index %d)", incident.id, level.name, idx)
            # set cursor default
            cursor = self._cursors.setdefault(self.policy.name + ":" + level.name, 0)
            targets = level.targets or []
            if not targets:
                logger.warning("Level %s has no targets; skipping", level.name)
                continue

            # choose target round-robin
            target = targets[cursor % len(targets)]
            self._cursors[self.policy.name + ":" + level.name] = (cursor + 1) % max(1, len(targets))

            channel = self._build_channel(target)
            if not channel:
                logger.error("No channel created for target: %s", target)
                self.persistence.log_audit(incident.id, f"no-channel for target: {target}")
                continue

            # attempt notification with retries
            max_retries = max(1, getattr(level, "retries", 1))
            backoff_conf = getattr(level, "retry_backoff", None)
            backoff = self.retry_backoff
            if isinstance(backoff_conf, dict):
                backoff = RetryBackoff(
                    base=backoff_conf.get("base", backoff.base),
                    factor=backoff_conf.get("factor", backoff.factor),
                    max_backoff=backoff_conf.get("max_backoff", backoff.max_backoff),
                    jitter=backoff_conf.get("jitter", backoff.jitter),
                )

            # Build the message to send; include metadata context
            formatted_message = f"[{self.policy.name}] {incident.title}\n{incident.message}\nID: {incident.id}"
            ok = self._notify_target(channel, formatted_message, max_retries, backoff)
            if ok:
                self.persistence.log_audit(incident.id, f"notified level:{level.name} target:{target}")
            else:
                self.persistence.log_audit(incident.id, f"failed notify level:{level.name} target:{target}")

            # If configured to stop on ack, and an external system may ACK the incident
            # this code checks the persisted incident status and stops if not open.
            latest_incident = self.persistence.get_incident(incident.id)
            if latest_incident and latest_incident.status != IncidentStatus.OPEN and stop_on_ack:
                logger.info("Incident %s acknowledged or resolved; stopping escalation", incident.id)
                return

            # wait the timeout before escalating to next level
            timeout_seconds = int(level.timeout.total_seconds()) if level.timeout else 0
            if timeout_seconds > 0:
                logger.debug("Sleeping for level timeout: %d seconds", timeout_seconds)
                time.sleep(timeout_seconds)

        logger.info("Escalation completed for incident %s (policy exhausted)", incident.id)
        self.persistence.log_audit(incident.id, "escalation:policy_exhausted")
