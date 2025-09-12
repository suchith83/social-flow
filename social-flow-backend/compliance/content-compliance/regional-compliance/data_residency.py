"""
data_residency.py

Utilities to enforce data residency and routing rules.
Includes:
- tag_data_for_region: attach region metadata to objects
- choose_storage_location: choose a storage region given rules and available backends
- migration recommendations
"""

from typing import List, Optional, Dict
import random
import logging

logger = logging.getLogger(__name__)

# Example available storage backends (region code -> endpoint)
_AVAILABLE_STORES = {
    "us": "s3-us.example.com",
    "eu": "s3-eu.example.com",
    "india": "s3-in.example.com",
    "global": "s3-global.example.com"
}

def tag_data_for_region(data_obj: Dict, target_region: str) -> None:
    """Add region tag metadata to data object (in-place)."""
    data_obj.setdefault("meta", {})["storage_region"] = target_region

def choose_storage_location(preferred_regions: List[str], fallback: Optional[str] = "global") -> str:
    """
    Choose an available storage backend given preferences and availability.
    Very simplistic: returns first match in _AVAILABLE_STORES; otherwise fallback.
    """
    for r in preferred_regions:
        if r in _AVAILABLE_STORES:
            logger.debug(f"Selected storage region {r}")
            return r
    logger.debug(f"No preferred storage available, falling back to {fallback}")
    return fallback

def recommend_migration(current_region: str, required_region: str) -> Dict[str, str]:
    """
    Provide a recommended migration plan (high-level):
    - where to copy data
    - suggested steps (stub)
    """
    if current_region.lower() == required_region.lower():
        return {"action": "none", "reason": "already compliant"}
    # choose target endpoint
    endpoint = _AVAILABLE_STORES.get(required_region.lower(), _AVAILABLE_STORES["global"])
    plan = {
        "action": "migrate",
        "target_region": required_region,
        "target_endpoint": endpoint,
        "steps": "snapshot -> copy -> re-point references -> verify -> delete-old"
    }
    return plan
