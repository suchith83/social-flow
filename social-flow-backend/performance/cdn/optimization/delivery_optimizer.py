# Optimizes delivery pipelines and resource allocation
# performance/cdn/optimization/delivery_optimizer.py
"""
Delivery optimization for streaming and protocol-level tweaks.
"""

import random
from typing import Dict, List
from .utils import logger

class DeliveryOptimizer:
    def __init__(self):
        self.protocols = ["http1.1", "http2", "quic"]

    def choose_protocol(self, client_supports: List[str]) -> str:
        """Pick best protocol supported by client."""
        for p in ["quic", "http2", "http1.1"]:
            if p in client_supports:
                logger.debug(f"Selected protocol {p}")
                return p
        return "http1.1"

    def adaptive_bitrate(self, bitrates: List[int], bandwidth_kbps: int) -> int:
        """Select bitrate closest to bandwidth capacity."""
        chosen = max([b for b in bitrates if b <= bandwidth_kbps], default=min(bitrates))
        logger.debug(f"Adaptive bitrate {chosen} for {bandwidth_kbps}kbps")
        return chosen

    def optimize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Optimize headers (strip unused, add caching hints)."""
        optimized = {k: v for k, v in headers.items() if k.lower() not in ["x-debug", "server"]}
        optimized["Cache-Control"] = "public, max-age=300"
        optimized["Connection"] = "keep-alive"
        return optimized
