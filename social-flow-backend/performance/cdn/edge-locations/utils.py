# Shared helper functions
# performance/cdn/edge-locations/utils.py
"""
Shared utilities for edge-locations.

Contains:
- dataclasses/types used across modules
- backoff utility
- simple asynchronous worker helpers
- logging config
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import logging
import time
import asyncio
import random
import ipaddress

# Logging configured for package
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("cdn.edge_locations")

@dataclass
class EdgeNode:
    id: str
    hostname: str
    ip: str
    region: str
    country: str
    city: Optional[str]
    capacity_rps: int  # requests per second capacity
    current_load_rps: float = 0.0
    tags: Optional[List[str]] = None
    healthy: bool = True
    last_seen: Optional[str] = None  # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def is_valid_ip(ip: str) -> bool:
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def jitter_backoff(base: float = 0.5, factor: float = 2.0, max_backoff: float = 30.0, attempts: int = 1) -> float:
    """Return a jittered backoff time in seconds."""
    exp = base * (factor ** (attempts - 1))
    capped = min(exp, max_backoff)
    return capped * (0.5 + random.random() * 0.5)

async def sleep_backoff(attempts: int, base: float = 0.5):
    delay = jitter_backoff(base=base, attempts=attempts)
    logger.debug(f"Sleeping for {delay:.2f}s (attempt {attempts})")
    await asyncio.sleep(delay)
