# Handles suppression over time windows
"""
Windowed suppression helpers.

A WindowedSuppressionRule suppresses repeated noisy alerts by key for a duration
once a threshold of occurrences in a sliding window is reached.

Example use-case:
  - If the same error occurs > 10 times in 30 seconds for host X, suppress further alerts for 5 minutes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
import threading
import logging

from .utils import now_utc, extract_path

logger = logging.getLogger(__name__)


@dataclass
class WindowedSuppressionRule:
    """
    Config for windowed suppression.

    Attributes:
      - name: friendly name
      - key_path: dotted path used to compute a key for grouping occurrences (e.g., "host" or "error.code")
      - count_threshold: number of occurrences in window required to trigger suppression
      - window_seconds: duration of sliding window to observe occurrences
      - suppress_duration_seconds: how long to suppress once triggered
      - scope: metadata or scope string
      - metadata: free-form dict
    """
    name: str
    key_path: str
    count_threshold: int
    window_seconds: int
    suppress_duration_seconds: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class WindowedSuppressor:
    """
    Maintains sliding windows per key and emits suppression directives when thresholds are exceeded.

    Implementation:
      - For each key, keep a deque of timestamps (UTC).
      - On each event, push timestamp and pop older than window.
      - If len(deque) >= count_threshold -> create a suppression entry (returned) and clear deque or mark suppressed.

    Note: This is in-memory. For distributed systems, persist counts in Redis sorted sets.
    """

    def __init__(self, rule: WindowedSuppressionRule):
        self.rule = rule
        self._windows: Dict[str, deque] = defaultdict(deque)  # key -> deque[timestamps]
        self._suppressed_until: Dict[str, datetime] = {}  # key -> expiry datetime
        self._lock = threading.RLock()

    def _current_ts(self):
        return now_utc()

    def record_event_and_maybe_trigger(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Record an event; if threshold exceeded, return a suppression descriptor dict:
          {"key": key, "suppress_until": datetime, "rule": self.rule}
        Otherwise return None.
        """
        with self._lock:
            key_val = extract_path(event, self.rule.key_path) or "__unknown__"
            key = str(key_val)
            now = self._current_ts()

            # If currently suppressed for key, return suppression info
            if key in self._suppressed_until and self._suppressed_until[key] > now:
                logger.debug("Key %s is already suppressed until %s", key, self._suppressed_until[key])
                return {
                    "key": key,
                    "suppress_until": self._suppressed_until[key],
                    "rule": self.rule,
                }

            dq = self._windows[key]
            dq.append(now)

            # purge older than window
            cutoff = now - timedelta(seconds=self.rule.window_seconds)
            while dq and dq[0] < cutoff:
                dq.popleft()

            logger.debug("Window length for key %s = %d", key, len(dq))

            if len(dq) >= self.rule.count_threshold:
                suppress_until = now + timedelta(seconds=self.rule.suppress_duration_seconds)
                self._suppressed_until[key] = suppress_until
                # clear window to avoid repeated immediate triggers
                dq.clear()
                logger.info("Windowed suppression triggered for key %s until %s", key, suppress_until)
                return {
                    "key": key,
                    "suppress_until": suppress_until,
                    "rule": self.rule,
                }
            return None

    def is_suppressed(self, event: Dict[str, Any]) -> bool:
        with self._lock:
            key_val = extract_path(event, self.rule.key_path) or "__unknown__"
            key = str(key_val)
            now = self._current_ts()
            return key in self._suppressed_until and self._suppressed_until[key] > now

    def cleanup_expired(self):
        """
        Remove expired suppressions and optionally stale windows to bound memory.
        """
        with self._lock:
            now = self._current_ts()
            expired = [k for k, v in self._suppressed_until.items() if v <= now]
            for k in expired:
                del self._suppressed_until[k]
            # prune windows older than window_seconds
            cutoff = now - timedelta(seconds=self.rule.window_seconds)
            for k, dq in list(self._windows.items()):
                while dq and dq[0] < cutoff:
                    dq.popleft()
                if not dq:
                    del self._windows[k]
