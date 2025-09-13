# Stores silenced events or alerts
"""
SilenceStore interface and an in-memory implementation.

This store persists suppression entries (silences). API:
 - save(rule: SuppressionRule)
 - get(rule_id) -> SuppressionRule | None
 - delete(rule_id)
 - list_active() -> List[SuppressionRule]
 - list_all() -> List[SuppressionRule]
"""

from typing import Optional, List, Dict
from threading import RLock
import logging

from .suppression_rule import SuppressionRule
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SilenceStore:
    """
    Interface-like base class. Implement concrete class backed by DB/Redis etc.
    """

    def save(self, rule: SuppressionRule) -> None:
        raise NotImplementedError()

    def get(self, rule_id: str) -> Optional[SuppressionRule]:
        raise NotImplementedError()

    def delete(self, rule_id: str) -> None:
        raise NotImplementedError()

    def list_active(self) -> List[SuppressionRule]:
        raise NotImplementedError()

    def list_all(self) -> List[SuppressionRule]:
        raise NotImplementedError()


class InMemorySilenceStore(SilenceStore):
    """
    Thread-safe in-memory store. Not suitable for multi-process deployments.
    """

    def __init__(self):
        self._store: Dict[str, SuppressionRule] = {}
        self._lock = RLock()

    def save(self, rule: SuppressionRule) -> None:
        with self._lock:
            self._store[rule.id] = rule
            logger.debug("Silence saved: %s (expires=%s)", rule.id, getattr(rule, "expires_at", None))

    def get(self, rule_id: str) -> Optional[SuppressionRule]:
        with self._lock:
            return self._store.get(rule_id)

    def delete(self, rule_id: str) -> None:
        with self._lock:
            self._store.pop(rule_id, None)
            logger.debug("Silence deleted: %s", rule_id)

    def list_active(self) -> List[SuppressionRule]:
        with self._lock:
            now = datetime.now(timezone.utc)
            out = []
            for r in self._store.values():
                if r.expires_at is None or r.expires_at > now:
                    out.append(r)
            return out

    def list_all(self) -> List[SuppressionRule]:
        with self._lock:
            return list(self._store.values())
