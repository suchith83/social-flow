# Dynamic load distribution
"""
Dynamic Load Balancer
Distributes workload across nodes adaptively.
"""

import random
import time
from typing import List
from .config import CONFIG


class LoadBalancer:
    def __init__(self, nodes: List[str] = None):
        self.nodes = nodes or [f"node-{i}" for i in range(1, CONFIG.max_nodes + 1)]
        self.last_rebalance = time.time()
        self.node_load = {node: 0 for node in self.nodes}

    def select_node(self) -> str:
        """Pick the node with the lowest load."""
        node = min(self.node_load, key=self.node_load.get)
        self.node_load[node] += 1
        return node

    def release_node(self, node: str):
        """Release load from a node."""
        if node in self.node_load and self.node_load[node] > 0:
            self.node_load[node] -= 1

    def rebalance(self):
        """Randomly redistribute loads periodically."""
        if time.time() - self.last_rebalance >= CONFIG.rebalance_interval_sec:
            total_load = sum(self.node_load.values())
            avg_load = total_load / len(self.node_load)
            for node in self.node_load:
                self.node_load[node] = int(avg_load + random.randint(-2, 2))
            self.last_rebalance = time.time()
