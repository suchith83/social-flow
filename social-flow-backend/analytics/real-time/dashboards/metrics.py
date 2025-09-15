from datetime import datetime
from typing import Dict, Any


def compute_kpis(events: list) -> Dict[str, Any]:
    """Compute Key Performance Indicators from event list."""
    return {
        "total_events": len(events),
        "last_updated": datetime.utcnow().isoformat(),
        "avg_value": sum(events) / len(events) if events else 0,
        "max_value": max(events) if events else 0,
        "min_value": min(events) if events else 0,
    }
