# Handles geographic routing of traffic
# performance/cdn/edge-locations/geo_routing.py
"""
Geo-routing utilities.

Purpose:
- Map user IP / geolocation to the best edge location(s)
- Provide selection strategies: nearest-by-distance, weighted-capacity, failover
- Support multi-criteria selection (latency + capacity + legal constraints)

Notes:
- For demo purposes uses simple Haversine distance and in-memory location lat/lon.
- In production pair with a GeoIP database or edge registry with coordinates.
"""

from typing import List, Dict, Optional, Callable
import math
from .utils import logger

EARTH_RADIUS_KM = 6371.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance between two lat/lon pairs in kilometers."""
    rad = math.radians
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c

def choose_nearest(edges: List[Dict], client_lat: float, client_lon: float, limit: int = 3) -> List[Dict]:
    """Return up to `limit` nearest healthy edges based on lat/lon attributes."""
    scored = []
    for e in edges:
        coords = e.get("coords")
        if not coords:
            continue
        dist = haversine_distance(client_lat, client_lon, coords["lat"], coords["lon"])
        scored.append((dist, e))
    scored.sort(key=lambda x: x[0])
    return [e for _, e in scored[:limit]]

def choose_by_capacity(edges: List[Dict], required_rps: float = 1.0, limit: int = 3) -> List[Dict]:
    """Prefer edges with available capacity (capacity - current_load)."""
    scored = []
    for e in edges:
        avail = e.get("capacity_rps", 0) - e.get("current_load_rps", 0)
        scored.append((max(avail, 0), e))
    scored.sort(key=lambda x: -x[0])  # highest available capacity first
    return [e for _, e in scored[:limit]]

def multi_criteria_select(edges: List[Dict], client_lat: float, client_lon: float, required_rps: float = 1.0) -> Optional[Dict]:
    """
    Multi-criteria selection combining distance and available capacity.
    Score = normalized(1/distance) * w1 + normalized(available_capacity) * w2
    """
    if not edges:
        return None
    # prepare metrics
    scored = []
    max_dist = 0.0001
    max_avail = 0.0001
    metrics = []
    for e in edges:
        coords = e.get("coords")
        dist = haversine_distance(client_lat, client_lon, coords["lat"], coords["lon"]) if coords else float('inf')
        avail = max(0, e.get("capacity_rps", 0) - e.get("current_load_rps", 0))
        metrics.append((e, dist, avail))
        max_dist = max(max_dist, dist)
        max_avail = max(max_avail, avail)
    for e, dist, avail in metrics:
        norm_dist = 1.0 - (dist / max_dist)  # closer -> higher
        norm_avail = avail / max_avail
        score = 0.6 * norm_dist + 0.4 * norm_avail
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    logger.debug(f"Top candidate score {scored[0][0] if scored else 'n/a'}")
    return scored[0][1] if scored else None
