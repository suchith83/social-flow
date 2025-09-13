# Actions executed when rules match
"""
Actions to run when rules match.

An action descriptor is a dict with at least:
  - "type": string (e.g. "notify", "create_incident", "silence")
  - "params": dict of params for the action

Action functions accept (ActionContext) and return ActionResult.

We include:
- notify_action: call channel(s) from monitoring.alerting.channels
- create_incident_action: create an Incident via escalation.Incident and persist it (via provided persistence)
- silence_action: mute a rule for a period

Actions are written to be easily testable and mockable.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging

from ..channels import BaseChannel  # type: ignore
from ..escalation import Incident, InMemoryPersistence  # type: ignore
from .templates import render_template

logger = logging.getLogger(__name__)

@dataclass
class ActionContext:
    """
    Context passed to actions.
    - rule_context: RuleMatchContext from rule.py
    - rule_store/persistence: optional stores for stateful ops
    - channels_factory: optional callable that builds channel instances from target descriptors
    """
    rule_context: "RuleMatchContext"
    stores: Dict[str, Any]  # e.g., {"rule_store": store, "inc_persistence": persistence}
    channels_factory: Optional[Any] = None


@dataclass
class ActionResult:
    success: bool
    info: Optional[Dict[str, Any]] = None


# helper to instantiate channels dynamically (kept simple — users can provide their own factory)
def _default_channel_factory(target: Dict[str, Any]) -> Optional[BaseChannel]:
    from ..channels import SlackChannel, EmailChannel, SMSChannel, WebhookChannel, TeamsChannel, PagerDutyChannel  # type: ignore
    t = target.get("type", "").lower()
    try:
        if t == "slack":
            return SlackChannel(webhook_url=target["webhook"])
        if t == "email":
            return EmailChannel(
                smtp_host=target["smtp_host"],
                smtp_port=target.get("smtp_port", 587),
                username=target.get("username"),
                password=target.get("password"),
                from_addr=target.get("from_addr"),
                to_addrs=target.get("to_addrs", []),
                use_tls=target.get("use_tls", True),
            )
        if t == "sms":
            return SMSChannel(
                account_sid=target["account_sid"],
                auth_token=target["auth_token"],
                from_number=target["from_number"],
                to_numbers=target.get("to_numbers", []),
            )
        if t == "webhook":
            return WebhookChannel(url=target["url"], headers=target.get("headers"))
        if t == "teams":
            return TeamsChannel(webhook_url=target["webhook"])
        if t == "pagerduty":
            return PagerDutyChannel(routing_key=target["routing_key"], severity=target.get("severity", "critical"))
    except Exception as e:
        logger.exception("Failed to build channel: %s", e)
    logger.warning("Unsupported channel target: %s", t)
    return None


def notify_action(action_descriptor: Dict[str, Any], ctx: ActionContext) -> ActionResult:
    """
    action_descriptor example:
      {"type":"notify", "params": {"targets":[{...}, {...}], "template":"...", "subject":"..."}}
    """
    params = action_descriptor.get("params", {})
    targets = params.get("targets", [])
    template = params.get("template")
    subject = params.get("subject", None)
    factory = ctx.channels_factory or _default_channel_factory

    rendered = render_template(template, ctx.rule_context.rule, ctx.rule_context.event, ctx.rule_context.matched_values)

    # notify all configured targets; report partial successes
    results = []
    for t in targets:
        channel = factory(t)
        if not channel:
            logger.error("No channel for target %s", t)
            results.append({"target": t, "success": False, "error": "no_channel"})
            continue
        try:
            ok = channel.send_alert(rendered, subject=subject)
            results.append({"target": t, "success": bool(ok)})
        except Exception as e:
            logger.exception("notify_action failed for %s: %s", t, e)
            results.append({"target": t, "success": False, "error": str(e)})

    success = all(r.get("success") for r in results) if results else False
    return ActionResult(success=success, info={"results": results})


def create_incident_action(action_descriptor: Dict[str, Any], ctx: ActionContext) -> ActionResult:
    """
    Create an Incident in escalation subsystem and persist it.
    action_descriptor example:
      {"type":"create_incident", "params": {"severity":"critical", "assign_policy":"critical-db"}}
    """
    params = action_descriptor.get("params", {})
    sev = params.get("severity", "critical")
    title_template = params.get("title_template") or "{{rule.name}}: {{event.alert_name or 'alert'}}"
    message_template = params.get("message_template") or "{{event | tojson}}"

    title = render_template(title_template, ctx.rule_context.rule, ctx.rule_context.event, ctx.rule_context.matched_values)
    message = render_template(message_template, ctx.rule_context.rule, ctx.rule_context.event, ctx.rule_context.matched_values)

    # Build incident object
    incident = Incident(title=title, message=message, metadata={"severity": sev, "rule_id": ctx.rule_context.rule.id})

    persistence = ctx.stores.get("inc_persistence")
    if persistence is None:
        # fallback to in-memory persistence (useful for tests)
        persistence = InMemoryPersistence()
    try:
        persistence.save_incident(incident)
        return ActionResult(success=True, info={"incident_id": incident.id})
    except Exception as e:
        logger.exception("Failed to create incident: %s", e)
        return ActionResult(success=False, info={"error": str(e)})


def silence_action(action_descriptor: Dict[str, Any], ctx: ActionContext) -> ActionResult:
    """
    Silence (mute) the rule for a given duration.
    action_descriptor example:
      {"type":"silence", "params": {"duration_seconds": 600}}
    """
    params = action_descriptor.get("params", {})
    dur = params.get("duration_seconds", 0)
    from datetime import datetime, timezone, timedelta
    rule = ctx.rule_context.rule
    rule.mute_until = datetime.now(timezone.utc) + timedelta(seconds=int(dur))
    store = ctx.stores.get("rule_store")
    if store:
        try:
            store.update_rule(rule)
        except Exception as e:
            logger.exception("Failed to persist silence for rule %s: %s", rule.id, e)
            return ActionResult(success=False, info={"error": str(e)})
    return ActionResult(success=True, info={"mute_until": rule.mute_until.isoformat()})
