# Core suppression logic
"""
Suppressor engine.

Responsibilities:
- maintain active suppression entries using a Silences store (persistence)
- evaluate inbound events and decide whether to suppress them
- create ad-hoc silences (persisted) and remove or cleanup expired ones
- provide auditing via optional logger/store
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import threading
import logging

from .suppression_rule import SuppressionRule, SuppressionScope
from .silence_store import SilenceStore, InMemorySilenceStore
from .utils import now_utc, extract_path

logger = logging.getLogger(__name__)


class SuppressionDecision:
    """
    Represents the decision for an event: suppressed or allowed.
    If suppressed, contains the matching suppression rule and reason.
    """
    def __init__(self, suppressed: bool, rule: Optional[SuppressionRule] = None, info: Optional[Dict[str, Any]] = None):
        self.suppressed = suppressed
        self.rule = rule
        self.info = info or {}

    def to_dict(self):
        return {
            "suppressed": self.suppressed,
            "rule_id": self.rule.id if self.rule else None,
            "rule_name": self.rule.name if self.rule else None,
            "info": self.info,
        }


class Suppressor:
    """
    High-level suppression manager.

    Usage:
      suppressor = Suppressor(store=InMemorySilenceStore())
      decision = suppressor.should_suppress(event)
      if decision.suppressed:
          drop or annotate the alert

    Notes:
    - Matching is intentionally straightforward: checks all active suppressions and returns the first match.
    - For high-performance use migrate store to Redis and use indexed selectors.
    """

    def __init__(self, store: Optional[SilenceStore] = None, cleanup_interval_seconds: int = 60):
        self.store = store or InMemorySilenceStore()
        self._lock = threading.RLock()
        self._cleanup_interval_seconds = cleanup_interval_seconds
        # optionally, start a background cleanup thread if desired in long-running process
        self._bg_thread = None
        self._stop_bg = threading.Event()

    def start_background_cleanup(self):
        """
        Starts a background thread that periodically expires old silences.
        Call stop_background_cleanup() to stop.
        """
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_bg.clear()
        self._bg_thread = threading.Thread(target=self._background_cleaner, daemon=True)
        self._bg_thread.start()

    def stop_background_cleanup(self):
        if self._bg_thread and self._bg_thread.is_alive():
            self._stop_bg.set()
            self._bg_thread.join(timeout=2.0)

    def _background_cleaner(self):
        import time
        while not self._stop_bg.is_set():
            try:
                self.cleanup_expired()
            except Exception:
                logger.exception("Background cleanup error")
            time.sleep(self._cleanup_interval_seconds)

    def add_suppression(self, rule: SuppressionRule) -> None:
        """
        Persist a suppression entry.
        """
        with self._lock:
            self.store.save(rule)
            logger.info("Added suppression %s (expires=%s)", rule.name, getattr(rule, "expires_at", None))

    def remove_suppression(self, rule_id: str) -> None:
        with self._lock:
            self.store.delete(rule_id)
            logger.info("Removed suppression %s", rule_id)

    def cleanup_expired(self) -> List[str]:
        """
        Remove expired suppression rules and return their ids.
        """
        with self._lock:
            expired = []
            for r in self.store.list_active():
                if r.is_expired():
                    self.store.delete(r.id)
                    expired.append(r.id)
                    logger.debug("Expired suppression removed: %s", r.id)
            return expired

    def should_suppress(self, event: Dict[str, Any], scope_hint: Optional[SuppressionScope] = None) -> SuppressionDecision:
        """
        Evaluate event against active suppression rules.

        Returns:
            SuppressionDecision(suppressed: bool, rule: SuppressionRule | None, info)
        """
        with self._lock:
            for rule in self.store.list_active():
                # optional scope hint: check quick mismatch
                if scope_hint and rule.scope != scope_hint and rule.scope != SuppressionScope.GLOBAL:
                    continue
                # compute if rule matches event
                try:
                    if rule.is_expired():
                        # will be cleaned up soon
                        continue
                    if rule.matches_event(event):
                        info = {
                            "matched_selector": rule.selector,
                            "expires_at": getattr(rule, "expires_at", None),
                        }
                        logger.debug("Event suppressed by rule %s", rule.name)
                        return SuppressionDecision(True, rule, info)
                except Exception:
                    logger.exception("Error matching suppression rule %s", rule.id)
            return SuppressionDecision(False, None, {})
