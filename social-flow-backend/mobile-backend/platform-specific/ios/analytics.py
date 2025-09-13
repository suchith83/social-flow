# iOS analytics ingestion + schema validation
"""
iOS analytics ingestion.

- Defines a Pydantic schema for iOS events (screen_view, interaction, crash)
- Validates and normalizes payloads
- Applies rate limiting and basic enrichment (device lookup, timestamp normalization)
- Publishes to a downstream queue (simulated here by storing into InMemoryStore)

Extend points:
 - Add signed events or HMAC validation
 - Push to Kafka / PubSub / Kinesis for processing
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import time
import asyncio

from .utils import rate_limiter, store
from .device_registry import iOSDeviceRegistry

class iOSEvent(BaseModel):
    event_type: str = Field(..., description="Event type: screen_view, tap, crash, custom")
    timestamp_ms: Optional[int] = None
    device_id: str
    bundle_id: Optional[str] = None
    app_version: Optional[str] = None
    payload: Optional[Dict[str, Any]] = {}

    @validator("timestamp_ms", pre=True, always=True)
    def set_ts(cls, v):
        return int(v) if v else int(time.time() * 1000)


class AnalyticsIngestor:
    def __init__(self, device_registry: iOSDeviceRegistry):
        self.registry = device_registry

    async def ingest(self, raw_event: Dict) -> Dict:
        """
        Validate and store event. Rate-limited per device_id.
        Returns normalized event and status.
        """
        event = iOSEvent(**raw_event)
        key = f"analytics:{event.device_id}"
        if not rate_limiter.allow(event.device_id):
            return {"ok": False, "reason": "rate_limited"}

        # enrichment: attach device record if available
        device = self.registry.get(event.device_id)
        out = event.dict()
        out["device"] = device
        # push to downstream store — replace by async publish to messaging
        q_key = f"events:{event.device_id}"
        existing = store.get(q_key) or []
        existing.append(out)
        store.set(q_key, existing)
        return {"ok": True, "event": out}
