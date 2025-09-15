import json
import statistics
from collections import deque
from typing import Dict, Any
from .utils import get_logger

logger = get_logger(__name__)


class SlidingWindowAggregator:
    """Maintains a sliding window of events and computes statistics."""

    def __init__(self, window_size: int = 100):
        self.window = deque(maxlen=window_size)

    def add_event(self, event: str) -> Dict[str, Any]:
        """Add new event and return updated metrics."""
        try:
            data = json.loads(event)
            value = data.get("value", 0)
            self.window.append(value)
            metrics = {
                "count": len(self.window),
                "mean": statistics.mean(self.window) if self.window else 0,
                "stdev": statistics.pstdev(self.window) if len(self.window) > 1 else 0,
            }
            logger.debug(f"Updated metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error in aggregator: {e}")
            return {}
