# Cleans, transforms, aggregates data
# performance/cdn/analytics/processor.py
"""
Processor module
================
Processes raw CDN analytics data:
- Cleansing
- Normalization
- Aggregation
- Transformation for storage or analysis
"""

from typing import Dict, List
import statistics
from .utils import logger

class CDNProcessor:
    def __init__(self):
        self.buffer: List[Dict] = []

    def clean(self, record: Dict) -> Dict:
        """Remove unwanted fields and normalize structure."""
        if "error" in record:
            return {}
        return {
            "endpoint": record.get("endpoint"),
            "timestamp": record.get("timestamp"),
            "latency_ms": record["data"].get("latency", None),
            "status_code": record["data"].get("status", None),
            "cache_hit": record["data"].get("cache_hit", False),
        }

    def aggregate(self, batch: List[Dict]) -> Dict:
        """Aggregate a batch of cleaned records."""
        latencies = [r["latency_ms"] for r in batch if r.get("latency_ms")]
        status_codes = [r["status_code"] for r in batch if r.get("status_code")]

        return {
            "count": len(batch),
            "avg_latency": statistics.mean(latencies) if latencies else None,
            "error_rate": status_codes.count(500) / len(status_codes) if status_codes else 0,
            "cache_hit_rate": sum(1 for r in batch if r.get("cache_hit")) / len(batch) if batch else 0,
        }

    def process(self, record: Dict) -> Dict:
        """Process a single record and return a clean version."""
        clean = self.clean(record)
        if clean:
            self.buffer.append(clean)
        return clean
