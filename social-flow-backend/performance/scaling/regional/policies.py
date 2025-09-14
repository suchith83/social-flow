# Regional traffic distribution policies
# performance/scaling/regional/policies.py

import logging
import random
from typing import Dict
from .exceptions import PolicyError


logger = logging.getLogger("regional.policies")


class LatencyPolicy:
    """
    Chooses region with lowest latency among healthy ones.
    """

    def select(self, regions: Dict[str, Dict]) -> str:
        healthy = {r: d for r, d in regions.items() if d["healthy"]}
        if not healthy:
            raise PolicyError("No healthy regions available")
        return min(healthy, key=lambda r: healthy[r]["latency"])


class WeightedPolicy:
    """
    Chooses region based on weights (probabilistic).
    """

    def select(self, regions: Dict[str, Dict]) -> str:
        healthy = [(r, d["weight"]) for r, d in regions.items() if d["healthy"]]
        if not healthy:
            raise PolicyError("No healthy regions available")
        regions_list, weights = zip(*healthy)
        return random.choices(regions_list, weights=weights, k=1)[0]


class FailoverPolicy:
    """
    Chooses first healthy region from a predefined priority list.
    """

    def __init__(self, priority: list):
        self.priority = priority

    def select(self, regions: Dict[str, Dict]) -> str:
        for r in self.priority:
            if regions.get(r, {}).get("healthy", False):
                return r
        raise PolicyError("No healthy regions in failover chain")


class GeoPolicy:
    """
    Chooses region based on client geo (fake: map IP to region).
    """

    def select(self, regions: Dict[str, Dict], client_ip: str) -> str:
        healthy = [r for r, d in regions.items() if d["healthy"]]
        if not healthy:
            raise PolicyError("No healthy regions available")
        # Fake: use last octet of IP to choose
        idx = int(client_ip.split(".")[-1]) % len(healthy)
        return healthy[idx]
