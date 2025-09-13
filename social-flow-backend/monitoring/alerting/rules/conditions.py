# Condition definitions for rules
"""
Condition primitives for rules.

We provide:
- Base Condition class
- Logical combinators: AndCondition, OrCondition, NotCondition (Not implicit via inverse)
- ComparisonCondition for numeric/string comparisons (>, <, ==, !=, >=, <=)
- ExistsCondition to check presence of a path in event
- Path extraction uses dotted path (e.g., "metrics.cpu.usage") with safe getters
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List
import operator
import logging

logger = logging.getLogger(__name__)

Comparator = Any

def _safe_get(event: Dict[str, Any], path: str):
    """Safely get nested value by dotted path. Returns (found, value)."""
    parts = path.split(".") if path else []
    cur = event
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return False, None
    return True, cur

class Condition:
    """
    Base Condition interface.
    Implement evaluate(event) -> (bool, extracted_values)
    extracted_values can be used by actions/templates.
    """

    def evaluate(self, event: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        raise NotImplementedError()


@dataclass
class ComparisonCondition(Condition):
    """
    Compare a numeric or string field extracted by path with a value using operator.

    operator: one of '>', '<', '>=', '<=', '==', '!=' or a callable comparator
    path: dotted path into event dict
    value: value to compare against (number or string)
    """
    path: str
    operator: Any  # str or callable
    value: Any

    _ops_map = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def evaluate(self, event: Dict[str, Any]):
        found, val = _safe_get(event, self.path)
        if not found:
            logger.debug("ComparisonCondition: path %s not found", self.path)
            return False, {}
        comp = None
        if callable(self.operator):
            comp = self.operator
        else:
            comp = self._ops_map.get(str(self.operator))
            if comp is None:
                raise ValueError(f"Operator {self.operator} not supported")
        try:
            ok = comp(val, self.value)
            return ok, {self.path: val} if ok else {}
        except Exception as e:
            logger.exception("Comparison evaluation failed: %s", e)
            return False, {}


@dataclass
class ExistsCondition(Condition):
    """
    Succeeds if a dotted path exists in event and optionally equals a given value.
    """
    path: str
    equals: Any = None
    require_equals: bool = False

    def evaluate(self, event: Dict[str, Any]):
        found, val = _safe_get(event, self.path)
        if not found:
            return False, {}
        if self.require_equals:
            ok = val == self.equals
            return ok, {self.path: val} if ok else {}
        return True, {self.path: val}


@dataclass
class AndCondition(Condition):
    conditions: List[Condition] = field(default_factory=list)

    def evaluate(self, event: Dict[str, Any]):
        accumulated = {}
        for c in self.conditions:
            ok, vals = c.evaluate(event)
            if not ok:
                return False, {}
            accumulated.update(vals)
        return True, accumulated


@dataclass
class OrCondition(Condition):
    conditions: List[Condition] = field(default_factory=list)

    def evaluate(self, event: Dict[str, Any]):
        for c in self.conditions:
            ok, vals = c.evaluate(event)
            if ok:
                return True, vals
        return False, {}
