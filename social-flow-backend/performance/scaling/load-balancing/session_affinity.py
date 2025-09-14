# Sticky sessions & consistent hashing
# performance/scaling/load_balancing/session_affinity.py

import logging
import hashlib
from typing import Dict
from .exceptions import AlgorithmError


logger = logging.getLogger("load_balancing.session_affinity")


class StickySession:
    """
    Session affinity using consistent hashing.
    """

    def __init__(self, nodes: Dict[str, Dict]):
        self.nodes = list(nodes.keys())

    def select(self, client_ip: str) -> str:
        if not self.nodes:
            raise AlgorithmError("No nodes available for sticky session")
        h = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return self.nodes[h % len(self.nodes)]
