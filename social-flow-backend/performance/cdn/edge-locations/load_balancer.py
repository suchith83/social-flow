# Implements load balancing strategies
# performance/cdn/edge-locations/load_balancer.py
"""
Load balancer integration helpers

- Basic algorithms for distributing traffic among chosen edges
- Weighted Round Robin, Least Connections, Hash-based sticky sessions
- Helpers to produce LB config snippets (e.g., for NGINX, Envoy, or cloud LB)
"""

from typing import List, Dict, Iterator, Tuple, Optional
import itertools
from .utils import logger

class WeightedRoundRobin:
    """
    Weighted Round Robin generator.
    edges: list of dicts with 'id' and 'weight' keys.
    Example usage:
        wrr = WeightedRoundRobin(edges)
        next(wrr)  # returns next edge dict
    """
    def __init__(self, edges: List[Dict]):
        # normalize weights
        self.pool: List[Tuple[int, Dict]] = []
        for e in edges:
            weight = int(e.get("weight", 1))
            self.pool.append((weight, e))
        self._iterator = self._create_cycle()

    def _create_cycle(self) -> Iterator[Dict]:
        while True:
            for weight, edge in self.pool:
                for _ in range(max(1, weight)):
                    yield edge

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        return next(self._iterator)

def least_loaded(edges: List[Dict], count: int = 1) -> List[Dict]:
    """Return `count` edges with lowest current_load_rps / capacity ratio."""
    scored = []
    for e in edges:
        cap = e.get("capacity_rps", 1)
        load = e.get("current_load_rps", 0.0)
        ratio = (load / cap) if cap else float('inf')
        scored.append((ratio, e))
    scored.sort(key=lambda x: x[0])
    return [e for _, e in scored[:count]]

def generate_nginx_upstream(name: str, edges: List[Dict]) -> str:
    """
    Generate a simple nginx upstream block.
    Each edge must have 'hostname' or 'ip', 'port' optional and 'weight' optional.
    """
    lines = [f"upstream {name} {{"]
    for e in edges:
        host = e.get("hostname") or e.get("ip")
        port = e.get("port", 80)
        weight = e.get("weight", 1)
        lines.append(f"    server {host}:{port} weight={weight};")
    lines.append("}")
    conf = "\n".join(lines)
    logger.debug(f"Generated nginx upstream for {name}")
    return conf
