# Lightweight FCM wrapper / push token manager
"""
Lightweight FCM wrapper and push token manager.

This file provides:
 - register / unregister device push tokens (ties into DeviceRegistry)
 - send push notifications (batching + retries)
 - abstraction so you can swap to other providers

Note: For demo we don't call real FCM. In production use `aiohttp` or `httpx` to POST to FCM
with an authenticated server key or OAuth2 credentials.
"""

import asyncio
import json
from typing import List, Dict, Optional
from .config import CONFIG
from .device_registry import DeviceRegistry
from .utils import push_rate_limiter

# For demo we simulate network calls
async def _post_to_fcm(batch_payload: Dict):
    """Simulated network call to FCM endpoint (replace with httpx async POST)."""
    await asyncio.sleep(0.05)  # simulate latency
    # return simulated response
    return {"success": True, "sent": len(batch_payload.get("registration_ids", []))}


class FCMManager:
    def __init__(self, device_registry: DeviceRegistry):
        self.registry = device_registry

    def register_token(self, raw_device_id: str, token: str, token_type: str = "fcm") -> Dict:
        """Persist push token on the device record."""
        meta = {"push_token": token, "push_token_type": token_type}
        return self.registry.register(raw_device_id, meta)

    def unregister_token(self, raw_device_id: str) -> bool:
        rec = self.registry.get(raw_device_id)
        if not rec:
            return False
        rec.pop("push_token", None)
        rec.pop("push_token_type", None)
        # update timestamp
        return bool(self.registry.register(raw_device_id, rec))

    async def send_multicast(self, tokens: List[str], payload: Dict) -> Dict:
        """
        Send to up to CONFIG.fcm_batch_size tokens per batch; uses simple rate limiting.
        Returns combined results. Replace with real FCM HTTP requests + error handling.
        """
        results = {"batches": []}
        # chunk tokens
        chunk_size = CONFIG.fcm_batch_size
        for i in range(0, len(tokens), chunk_size):
            batch = tokens[i : i + chunk_size]
            # rate limit check
            if not push_rate_limiter.allow("global"):
                results["batches"].append({"error": "rate_limited"})
                await asyncio.sleep(0.2)
                continue
            payload_batch = {"registration_ids": batch, "data": payload}
            resp = await _post_to_fcm(payload_batch)
            results["batches"].append(resp)
        return results

    async def notify_devices_by_query(self, min_android: int, message: Dict) -> Dict:
        """Send notifications to devices filtered by capability."""
        devices = self.registry.query_capability(min_android=min_android)
        tokens = [d.get("push_token") for d in devices if d.get("push_token")]
        # remove falsy
        tokens = [t for t in tokens if t]
        return await self.send_multicast(tokens, message)
