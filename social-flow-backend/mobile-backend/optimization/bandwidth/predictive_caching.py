# ML-driven caching of likely future requests
"""
Predictive Caching Module
Uses simple ML-like prediction (Markov chains / frequency analysis)
to prefetch likely needed resources.
"""

import random
from collections import defaultdict, deque
from typing import Dict, List
from .config import CONFIG


class PredictiveCache:
    def __init__(self):
        self.cache: Dict[str, bytes] = {}
        self.history = deque(maxlen=100)
        self.transition_prob = defaultdict(lambda: defaultdict(int))

    def record_access(self, resource_id: str) -> None:
        """Record access and update transition probabilities."""
        if self.history:
            prev = self.history[-1]
            self.transition_prob[prev][resource_id] += 1
        self.history.append(resource_id)

    def predict_next(self) -> str:
        """Predict the most likely next resource."""
        if not self.history:
            return None
        last = self.history[-1]
        candidates = self.transition_prob[last]
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def prefetch(self, fetch_fn) -> None:
        """Prefetch predicted resource using provided fetch function."""
        resource_id = self.predict_next()
        if resource_id and resource_id not in self.cache:
            self.cache[resource_id] = fetch_fn(resource_id)

    def get(self, resource_id: str, fetch_fn) -> bytes:
        """Get resource, fetching if not cached."""
        if resource_id not in self.cache:
            self.cache[resource_id] = fetch_fn(resource_id)
        self.record_access(resource_id)
        return self.cache[resource_id]
