# React Native analytics ingestion + schema validation
"""
Analytics ingestion for React Native.

 - Pydantic schema for common RN events (navigation, interaction, error)
 - Validation, rate limiting, enrichment with device info
 - Stores to InMemoryStore for demo (replace with message queue / datastore)
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import time
import asyncio

from .utils import rate_limiter, store
from .device_registry import RNDeviceRegistry

class RNEvent(BaseModel):
    event_type: str
    timestamp_ms: Optional[int] = None
    device_id: str
    category: Optional[str] = None
    payload: Optional[Dict[str, Any]] = {}

    @validator("timestamp_ms", pre=True, always=True)
    def set_ts(cls, v):
        return int(v) if v else int(time.time() * 1000)


class RNAnalyticsIngestor:
    def __init__(self, registry: RNDeviceRegistry):
        self.registry = registry

    async def ingest(self, raw_event: Dict) -> Dict:
        event = RNEvent(**raw_event)
        if not rate_limiter.allow(event.device_id):
            return {"ok": False, "reason": "rate_limited"}
        device = self.registry.get(event.device_id)
        out = event.dict()
        out["device"] = device
        q_key = f"rn_events:{event.device_id}"
        existing = store.get(q_key) or []
        existing.append(out)
        store.set(q_key, existing)
        return {"ok": True, "event": out}
