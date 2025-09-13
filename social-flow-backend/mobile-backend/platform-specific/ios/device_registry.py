# Device records and capability detection
"""
Device registry for iOS devices.

Responsibilities:
 - Register device metadata (model, iOS version, screen, device token)
 - Update APNs device token associations
 - Query devices for targeted notifications / rollouts
 - Prune stale device records

Uses InMemoryStore for demo; replace with persistent DB in production.
"""

import time
from typing import Dict, Optional, List
from .utils import store, device_hash
from .config import CONFIG

DEVICE_PREFIX = "ios_device:"


class iOSDeviceRegistry:
    def __init__(self):
        self._store = store

    def register(self, raw_device_id: str, payload: Dict) -> Dict:
        """
        Register or update a device.
        payload example:
        {
            "model": "iPhone12,1",
            "ios_version": "16.4",
            "screen": {"w":828, "h":1792, "scale":2},
            "device_token": "abcd...",
            "app_version": "1.2.3",
            "installed_bundle_sha": "..."
        }
        """
        did = device_hash(raw_device_id)
        key = DEVICE_PREFIX + did
        record = self._store.get(key) or {}
        record.update(payload)
        record["device_id"] = did
        record["updated_at"] = time.time()
        self._store.set(key, record)
        return record

    def get(self, raw_device_id: str) -> Optional[Dict]:
        did = device_hash(raw_device_id)
        return self._store.get(DEVICE_PREFIX + did)

    def query_by_capability(self, min_ios: Optional[str] = None, model: Optional[str] = None) -> List[Dict]:
        """Return devices matching capability filters for targeted rollouts."""
        out = []
        for k, v in self._store.scan(DEVICE_PREFIX).items():
            if model and v.get("model") != model:
                continue
            if min_ios:
                try:
                    # compare major.minor
                    def norm(s):
                        return tuple(map(int, (s or "0.0").split(".")[:2]))
                    if norm(v.get("ios_version")) < norm(min_ios):
                        continue
                except Exception:
                    pass
            out.append(v)
        return out

    def prune_stale(self, older_than_days: int = CONFIG.device_prune_days) -> int:
        cutoff = time.time() - older_than_days * 86400
        deleted = 0
        for k, v in list(self._store.scan(DEVICE_PREFIX).items()):
            if v.get("updated_at", 0) < cutoff:
                self._store.delete(k)
                deleted += 1
        return deleted
