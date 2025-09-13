# Push notification handling for React Native apps
"""
Push manager (FCM + APNs helpers) for React Native.

Features:
 - token registration/unregistration tied to RNDeviceRegistry
 - batch send helpers for FCM and APNs (simulated network calls)
 - simple rate limiting and token canonicalization

Replace simulated network calls with real HTTP clients (httpx, aiohttp) and platform SDKs in prod.
"""

import asyncio
from typing import List, Dict, Optional
from .device_registry import RNDeviceRegistry
from .utils import rate_limiter

# Simulated network functions
async def _fake_fcm_send(tokens: List[str], payload: Dict):
    await asyncio.sleep(0.05)
    # simulate success/fail per token
    return [{"token": t, "success": not t.endswith("invalid")} for t in tokens]

async def _fake_apns_send(tokens: List[str], payload: Dict):
    await asyncio.sleep(0.03)
    return [{"token": t, "success": not t.endswith("invalid")} for t in tokens]


class PushManager:
    def __init__(self, registry: RNDeviceRegistry):
        self.registry = registry

    def register_token(self, raw_device_id: str, token: str, platform: str = "android") -> Dict:
        return self.registry.register(raw_device_id, {"push_token": token, "platform": platform})

    def unregister_token(self, raw_device_id: str) -> bool:
        rec = self.registry.get(raw_device_id)
        if not rec:
            return False
        rec.pop("push_token", None)
        self.registry.register(raw_device_id, rec)
        return True

    async def send_multicast(self, platform: str, tokens: List[str], payload: Dict) -> Dict:
        """
        Send to tokens in batches; uses simplistic global rate limiter.
        """
        results = {"batches": []}
        if platform == "android":
            chunk = 1000
            for i in range(0, len(tokens), chunk):
                batch = tokens[i : i + chunk]
                if not rate_limiter.allow("fcm_global"):
                    results["batches"].append({"error": "rate_limited"})
                    await asyncio.sleep(0.1)
                    continue
                resp = await _fake_fcm_send(batch, payload)
                results["batches"].append(resp)
        elif platform == "ios":
            chunk = 500
            for i in range(0, len(tokens), chunk):
                batch = tokens[i : i + chunk]
                if not rate_limiter.allow("apns_global"):
                    results["batches"].append({"error": "rate_limited"})
                    await asyncio.sleep(0.1)
                    continue
                resp = await _fake_apns_send(batch, payload)
                results["batches"].append(resp)
        else:
            return {"error": "unsupported_platform"}
        return results

    async def notify_query(self, platform: Optional[str], message: Dict, min_rn: Optional[str] = None) -> Dict:
        devices = self.registry.query(platform=platform, min_rn=min_rn)
        tokens = [d.get("push_token") for d in devices if d.get("push_token")]
        tokens = [t for t in tokens if t]
        return await self.send_multicast(platform or "android", tokens, message)
