# Core rules engine to evaluate and trigger rules
"""
RuleEngine - evaluate events against rules and run actions.

Features:
- Evaluate enabled, unsilenced rules
- Optional priority ordering
- Supports synchronous execution and pluggable async executor
- Returns a summary of matched rules and action results
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timezone

from .persistence import RuleStore, InMemoryRuleStore
from .rule import RuleMatchContext, Rule
from .validator import validate_rule
from .actions import ActionContext, notify_action, create_incident_action, silence_action, ActionResult

logger = logging.getLogger(__name__)

# map action type -> function
_ACTION_REGISTRY = {
    "notify": notify_action,
    "create_incident": create_incident_action,
    "silence": silence_action,
}

class RuleEngine:
    def __init__(self, store: Optional[RuleStore] = None, async_executor=None, max_matches_per_event: int = 10):
        """
        Parameters:
          - store: persistence for rules (defaults to in-memory)
          - async_executor: optional executor(fn, args) -> schedule async task. If None, runs synchronously.
          - max_matches_per_event: safety limit
        """
        self.store = store or InMemoryRuleStore()
        self.async_executor = async_executor
        self.max_matches_per_event = max_matches_per_event

    def add_rule(self, rule: Rule):
        validate_rule(rule, store=self.store)
        self.store.add_rule(rule)

    def update_rule(self, rule: Rule):
        validate_rule(rule, store=self.store)
        rule.updated_at = datetime.now(timezone.utc)
        self.store.update_rule(rule)

    def delete_rule(self, rule_id: str):
        self.store.delete_rule(rule_id)

    def list_rules(self) -> List[Rule]:
        return self.store.list_rules()

    def evaluate(self, event: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate all rules against the event. Returns a result dict:
        {
          "matches": [
             {"rule_id": ..., "rule_name": ..., "actions": [ { "type":..., "result": ActionResult }, ... ] }
          ],
          "summary": {...}
        }
        """
        matches = []
        count = 0
        for rule in self.store.list_rules():
            if not rule.enabled:
                continue
            if rule.is_silenced():
                logger.debug("Rule %s is silenced; skipping", rule.id)
                continue

            try:
                ok, matched_values = rule.condition.evaluate(event)
            except Exception as e:
                logger.exception("Rule evaluation error for %s: %s", rule.name, e)
                continue

            if not ok:
                continue

            count += 1
            if count > self.max_matches_per_event:
                logger.warning("Reached max_matches_per_event=%d - stopping further evaluations", self.max_matches_per_event)
                break

            ctx = RuleMatchContext(event=event, matched_values=matched_values, rule=rule)
            action_ctx = ActionContext(rule_context=ctx, stores={"rule_store": self.store}, channels_factory=None,)

            action_results = []
            for action_desc in rule.actions:
                atype = action_desc.get("type")
                fn = _ACTION_REGISTRY.get(atype)
                if not fn:
                    logger.error("Unknown action type: %s", atype)
                    action_results.append({"type": atype, "result": ActionResult(success=False, info={"error":"unknown_action"})})
                    continue

                # run either sync or async depending on executor
                if self.async_executor:
                    # schedule for background; executor should accept (callable, *args, **kwargs)
                    try:
                        self.async_executor(fn, action_desc, action_ctx)
                        action_results.append({"type": atype, "result": ActionResult(success=True, info={"scheduled": True})})
                    except Exception as e:
                        logger.exception("Async scheduling failed: %s", e)
                        action_results.append({"type": atype, "result": ActionResult(success=False, info={"error": str(e)})})
                else:
                    try:
                        res = fn(action_desc, action_ctx)
                        action_results.append({"type": atype, "result": res})
                    except Exception as e:
                        logger.exception("Action execution failed: %s", e)
                        action_results.append({"type": atype, "result": ActionResult(success=False, info={"error": str(e)})})

            matches.append({
                "rule_id": rule.id,
                "rule_name": rule.name,
                "matched_values": matched_values,
                "actions": action_results,
            })

        return {"matches": matches, "summary": {"total_matches": len(matches)}}
