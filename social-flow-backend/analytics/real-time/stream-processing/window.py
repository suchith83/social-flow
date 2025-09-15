import time
from collections import deque
from typing import Dict, List
from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


class SlidingWindow:
    """Sliding window for stream events."""

    def __init__(self, size_sec: int = settings.window_size_sec, slide_sec: int = settings.slide_interval_sec):
        self.size_sec = size_sec
        self.slide_sec = slide_sec
        self.events = deque()

    def add(self, event: Dict) -> None:
        """Add event and expire old ones."""
        ts = event.get("timestamp", time.time())
        self.events.append((ts, event))
        self._expire(ts)

    def _expire(self, now: float) -> None:
        """Remove events outside window size."""
        while self.events and now - self.events[0][0] > self.size_sec:
            self.events.popleft()

    def get_events(self) -> List[Dict]:
        """Return current window events."""
        return [e for _, e in self.events]
