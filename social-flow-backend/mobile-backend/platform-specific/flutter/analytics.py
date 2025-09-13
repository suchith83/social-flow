# Flutter analytics ingestion + schema validation
"""
Analytics ingestion tailored for Flutter clients.

- Pydantic schema for Flutter events, including performance traces, error reports, and custom events
- Validation and enrichment with device registry
- Rate limiting per device
- Simple storage to InMemoryStore (replace with Kafka / PubSub in prod)
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import time
import asyncio

from .utils import rate_limiter, store
from .device_registry import FlutterDeviceRegistry

class FlutterEvent(BaseModel):
    event_type: str
    timestamp_ms: Optional[int] = None
    device_id: str
    category: Optional[str] = None
    payload: Optional[Dict[str, Any]] = {}

    @validator("timestamp_ms", pre=True, always=True)
    def set_ts(cls, v):
        return int(v) if v else int(time.time() * 1000)

class FlutterAnalyticsIngestor:
    def __init__(self, device_registry: FlutterDeviceRegistry):
        self.registry = device_registry

    async def ingest(self, raw_event: Dict) -> Dict:
        event = FlutterEvent(**raw_event)
        if not rate_limiter.allow(event.device_id):
            return {"ok": False, "reason": "rate_limited"}
        device = self.registry.get(event.device_id)
        out = event.dict()
        out["device"] = device
        q_key = f"flutter_events:{event.device_id}"
        existing = store.get(q_key) or []
        existing.append(out)
        store.set(q_key, existing)
        return {"ok": True, "event": out}
