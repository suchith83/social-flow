# Device records and capability detection
"""
Device registry tailored for React Native clients.

Tracks:
 - OS (android / ios), OS version
 - RN version, JS engine (Hermes / JSC)
 - patch / OTA installed bundle/version (for CodePush-like flows)
 - platform-specific push tokens
 - other capabilities (locale, device model, screen)

Uses InMemoryStore for demo; replace with DB in production.
"""

import time
from typing import Dict, Optional, List
from .utils import store, device_hash
from .config import CONFIG

DEVICE_PREFIX = "rn_device:"


class RNDeviceRegistry:
    def __init__(self):
        self._store = store

    def register(self, raw_device_id: str, payload: Dict) -> Dict:
        """
        Register / update a device.
        Example payload:
        {
           "platform": "android"|"ios",
           "os_version": "14.4",
           "rn_version": "0.72.0",
           "js_engine": "hermes"|"jsc",
           "app_version": "1.2.3",
           "installed_bundle_sha": "...",
           "push_token": "..."
        }
        """
        did = device_hash(raw_device_id)
        key = DEVICE_PREFIX + did
        rec = self._store.get(key) or {}
        rec.update(payload)
        rec["device_id"] = did
        rec["updated_at"] = time.time()
        self._store.set(key, rec)
        return rec

    def get(self, raw_device_id: str) -> Optional[Dict]:
        did = device_hash(raw_device_id)
        return self._store.get(DEVICE_PREFIX + did)

    def query(self, platform: Optional[str] = None, min_rn: Optional[str] = None, js_engine: Optional[str] = None) -> List[Dict]:
        """
        Query devices by capability for targeted rollouts.
        min_rn should be 'major.minor.patch' string; lexicographic semver compare naive.
        """
        out = []
        for k, v in self._store.scan(DEVICE_PREFIX).items():
            if platform and v.get("platform") != platform:
                continue
            if js_engine and v.get("js_engine") != js_engine:
                continue
            if min_rn:
                try:
                    def norm(s):
                        return tuple(map(int, (s or "0.0.0").split(".")[:3]))
                    if norm(v.get("rn_version")) < norm(min_rn):
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
