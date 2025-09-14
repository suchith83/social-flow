# Load balancing algorithms (round robin, least connections, consistent hashing, weighted)
# performance/scaling/load_balancing/algorithms.py

import itertools
import hashlib
import logging
from typing import Dict, List
from .exceptions import AlgorithmError


logger = logging.getLogger("load_balancing.algorithms")


class RoundRobin:
    def __init__(self, nodes: Dict[str, Dict]):
        self.nodes = list(nodes.keys())
        self._cycle = itertools.cycle(self.nodes)

    def select(self) -> str:
        return next(self._cycle)


class LeastConnections:
    def __init__(self, nodes: Dict[str, Dict]):
        self.nodes = nodes

    def select(self) -> str:
        try:
            return min(
                (nid for nid, n in self.nodes.items() if n["healthy"]),
                key=lambda nid: self.nodes[nid]["load"],
            )
        except ValueError:
            raise AlgorithmError("No healthy nodes available")


class WeightedRoundRobin:
    def __init__(self, nodes: Dict[str, Dict]):
        self.pool = []
        for nid, n in nodes.items():
            self.pool.extend([nid] * n.get("weight", 1))
        self._cycle = itertools.cycle(self.pool)

    def select(self) -> str:
        return next(self._cycle)


class ConsistentHashing:
    def __init__(self, nodes: Dict[str, Dict]):
        self.nodes = list(nodes.keys())

    def select(self, key: str) -> str:
        if not self.nodes:
            raise AlgorithmError("No nodes available for hashing")
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return self.nodes[h % len(self.nodes)]
