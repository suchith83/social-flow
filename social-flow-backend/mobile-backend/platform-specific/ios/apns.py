# Apple Push Notification Service wrapper
"""
APNs (Apple Push Notification service) helpers.

This implements:
 - device token registration helpers (ties into device registry)
 - batch send helpers that construct APNs requests (JWT-based) — simulated network calls here
 - token management stubs (canonicalization, invalidation)

Notes:
 - In production use an HTTP/2 client (httpx, hyper) and sign JWT with ES256 using your .p8 key (PyJWT or cryptography).
 - Handle feedback from APNs (unregistered device tokens) and remove tokens accordingly.
"""

import asyncio
from typing import List, Dict, Optional
from .device_registry import iOSDeviceRegistry
from .utils import build_apns_jwt, rate_limiter
from .config import CONFIG

# Demo network function (simulates APNs HTTP/2 POST)
async def _post_to_apns(token: str, payload: Dict, jwt: str):
    # simulate latency & response
    await asyncio.sleep(0.03)
    # simulate some tokens being invalid occasionally
    invalid = token.endswith("invalid")
    return {"success": not invalid, "token": token, "invalid": invalid}


class APNSManager:
    def __init__(self, device_registry: iOSDeviceRegistry):
        self.registry = device_registry
        # Precreate JWT for demo (in prod, rotate every ~20 minutes)
        self._jwt = build_apns_jwt(CONFIG.apns_key_id, CONFIG.apns_team_id, CONFIG.apns_auth_key_path)

    def register_token(self, raw_device_id: str, token: str) -> Dict:
        return self.registry.register(raw_device_id, {"device_token": token})

    def unregister_token(self, raw_device_id: str) -> bool:
        rec = self.registry.get(raw_device_id)
        if not rec:
            return False
        rec.pop("device_token", None)
        self.registry.register(raw_device_id, rec)
        return True

    async def send_multicast(self, tokens: List[str], payload: Dict) -> Dict:
        """
        Send notifications to a list of APNs device tokens.
        Uses simple rate limiting. In production use connection pooling and HTTP/2.
        """
        results = {"batches": []}
        chunk = CONFIG.apns_batch_size
        jwt = self._jwt or ""
        for i in range(0, len(tokens), chunk):
            batch = tokens[i : i + chunk]
            if not rate_limiter.allow("apns_global"):
                results["batches"].append({"error": "rate_limited"})
                await asyncio.sleep(0.1)
                continue
            batch_res = []
            # send concurrently to tokens in batch
            tasks = [asyncio.create_task(_post_to_apns(t, payload, jwt)) for t in batch]
            for t in tasks:
                r = await t
                batch_res.append(r)
            results["batches"].append(batch_res)
        return results

    async def notify_devices_by_query(self, min_ios: Optional[str], message: Dict) -> Dict:
        devices = self.registry.query_by_capability(min_ios=min_ios)
        tokens = [d.get("device_token") for d in devices if d.get("device_token")]
        tokens = [t for t in tokens if t]
        return await self.send_multicast(tokens, message)
