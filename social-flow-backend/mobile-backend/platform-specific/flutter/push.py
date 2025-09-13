# Push notification handling for Flutter apps
"""
Push manager for Flutter apps.

Supports:
 - FCM for Android (and iOS via FCM APNs fallback)
 - APNs wrapper placeholder (server keyless provider using p8 file)
 - Token registration/unregistration tied into device registry
 - Batch send / rate limiting / simulated network calls

In production:
 - Use Firebase Admin SDK for Python for FCM (async via httpx or aiohttp)
 - Use an APNs provider (apns2 or HTTP/2 provider) for direct iOS pushes
 - Handle token canonicalization and invalidation (feedback from responses)
"""

import asyncio
from typing import List, Dict, Optional
from .device_registry import FlutterDeviceRegistry
from .utils import rate_limiter
from .config import CONFIG

async def _fake_send(payload: Dict):
    await asyncio.sleep(0.03)
    return {"ok": True, "sent": len(payload.get("tokens", []))}

class PushManager:
    def __init__(self, device_registry: FlutterDeviceRegistry):
        self.registry = device_registry

    def register_token(self, raw_device_id: str, token: str, platform: str = "android") -> Dict:
        rec = self.registry.register(raw_device_id, {"push_token": token, "platform": platform})
        return rec

    def unregister_token(self, raw_device_id: str):
        rec = self.registry.get(raw_device_id)
        if not rec:
            return False
        rec.pop("push_token", None)
        self.registry.register(raw_device_id, rec)
        return True

    async def send_multicast(self, tokens: List[str], payload: Dict) -> Dict:
        """
        Batch tokens into CONFIG.fcm_batch_size and send asynchronously.
        Uses a simplistic rate limiter. Replace _fake_send with real FCM / APNs calls.
        """
        results = []
        chunk = CONFIG.fcm_batch_size
        for i in range(0, len(tokens), chunk):
            batch = tokens[i : i + chunk]
            if not rate_limiter.allow("push_global"):
                results.append({"error": "rate_limited"})
                await asyncio.sleep(0.1)
                continue
            resp = await _fake_send({"tokens": batch, "payload": payload})
            results.append(resp)
        return {"batches": results}

    async def notify_query(self, platform: Optional[str], message: Dict, min_flutter: Optional[str] = None) -> Dict:
        devices = self.registry.query_by_capability(min_flutter=min_flutter, platform=platform)
        tokens = [d.get("push_token") for d in devices if d.get("push_token")]
        tokens = [t for t in tokens if t]
        return await self.send_multicast(tokens, message)
