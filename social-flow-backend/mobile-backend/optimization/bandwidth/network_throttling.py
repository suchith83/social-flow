# Dynamically throttles bandwidth usage
"""
Network Throttling Module
Ensures fair bandwidth allocation per user/session.
"""

import time
from collections import defaultdict
from .config import CONFIG


class Throttler:
    def __init__(self):
        self.usage: Dict[str, List[float]] = defaultdict(list)

    def record_usage(self, user_id: str, kb_used: float):
        """Record network usage for a user."""
        now = time.time()
        self.usage[user_id].append((now, kb_used))
        # Clean up old entries
        self.usage[user_id] = [
            (t, kb) for (t, kb) in self.usage[user_id] if now - t < CONFIG.throttle_check_interval_sec
        ]

    def is_throttled(self, user_id: str) -> bool:
        """Check if user exceeded bandwidth cap."""
        total_kb = sum(kb for _, kb in self.usage[user_id])
        return total_kb > CONFIG.max_bandwidth_per_user_kbps * CONFIG.throttle_check_interval_sec / 8
