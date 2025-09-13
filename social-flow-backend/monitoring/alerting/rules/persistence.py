# Stores and retrieves rules from storage
"""
Simple rule store persistence.

Provides a RuleStore interface and an InMemoryRuleStore implementation.
In production, implement a RuleStore backed by Postgres/DynamoDB/Redis.
"""

from typing import Dict, Any, List, Optional
from threading import RLock
import logging
from .rule import Rule

logger = logging.getLogger(__name__)

class RuleStore:
    """
    Interface-like class. Implementations should provide:
      - add_rule(rule)
      - update_rule(rule)
      - delete_rule(rule_id)
      - list_rules()
      - get_rule(rule_id)
      - get_rule_by_name(name)
    """
    def add_rule(self, rule: Rule) -> None:
        raise NotImplementedError()
    def update_rule(self, rule: Rule) -> None:
        raise NotImplementedError()
    def delete_rule(self, rule_id: str) -> None:
        raise NotImplementedError()
    def list_rules(self) -> List[Rule]:
        raise NotImplementedError()
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        raise NotImplementedError()
    def get_rule_by_name(self, name: str) -> Optional[Rule]:
        raise NotImplementedError()


class InMemoryRuleStore(RuleStore):
    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._lock = RLock()

    def add_rule(self, rule: Rule) -> None:
        with self._lock:
            if rule.id in self._rules:
                raise KeyError("Rule id already exists")
            self._rules[rule.id] = rule

    def update_rule(self, rule: Rule) -> None:
        with self._lock:
            if rule.id not in self._rules:
                raise KeyError("Rule not found")
            self._rules[rule.id] = rule

    def delete_rule(self, rule_id: str) -> None:
        with self._lock:
            self._rules.pop(rule_id, None)

    def list_rules(self) -> List[Rule]:
        with self._lock:
            return list(self._rules.values())

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        with self._lock:
            return self._rules.get(rule_id)

    def get_rule_by_name(self, name: str) -> Optional[Rule]:
        with self._lock:
            for r in self._rules.values():
                if r.name == name:
                    return r
            return None
