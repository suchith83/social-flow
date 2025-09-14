# Shared helper functions
# performance/cdn/analytics/utils.py
"""
Utility functions shared across analytics modules.
"""

import logging
import json
import hashlib
import datetime
from typing import Dict, Any

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("cdn.analytics.utils")

def hash_key(data: str) -> str:
    """Generate a deterministic hash for partitioning or caching."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def now_utc() -> str:
    """Return the current UTC timestamp as ISO string."""
    return datetime.datetime.utcnow().isoformat()

def safe_json_dumps(data: Dict[str, Any]) -> str:
    """Safely convert a dict to JSON string with error handling."""
    try:
        return json.dumps(data, default=str)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        return "{}"
