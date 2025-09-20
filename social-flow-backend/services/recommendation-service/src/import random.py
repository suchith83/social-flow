import random
import time
from typing import Dict, Any


class InMemoryFeatureStore:
    """Small deterministic-ish feature generator for users for local dev."""

    def __init__(self, seed: int = 1234):
        self.seed = seed
        random.seed(self.seed)

    def user_features(self, user_id: str) -> Dict[str, Any]:
        """Return a small feature dict for a user_id."""
        # deterministic pseudo-random features derived from user_id
        h = sum(ord(c) for c in user_id) + int(time.time()) // 3600  # change slowly over time
        return {
            "user_id": user_id,
            "activity_score": (h % 100) / 100.0,
            "pref_genre_score": ((h >> 3) % 50) / 50.0,
            "recent_views": int((h % 10)),
        }
