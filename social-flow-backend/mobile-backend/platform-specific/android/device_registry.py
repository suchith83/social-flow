# Device records and capability detection
"""
Device registry & capability detection.

Responsibilities:
 - Register device metadata (model, android version, screen, codecs)
 - Update push token associations
 - Query capabilities for adaptive delivery (e.g., ABI, feature support)
 - Prune stale devices

This module uses the InMemoryStore in utils by default, but can be adapted to DB.
"""

import time
from typing import Dict, Optional, List
from .utils import store, device_hash, ensure_dir
from .config import CONFIG

DEVICE_PREFIX = "device:"  # key prefix in store


class DeviceRegistry:
    def __init__(self):
        # store is an InMemoryStore instance shared from utils
        self._store = store

    def register(self, raw_device_id: str, payload: Dict) -> Dict:
        """
        Register or update a device.
        payload example:
          {
            "model": "Pixel 7",
            "android_version": "13",
            "arch": "arm64-v8a",
            "screen": {"w":1080,"h":2340,"density":480},
            "supported_codecs": ["avc", "hevc"],
            "push_token": "abcd..."
          }
        """
        did = device_hash(raw_device_id)
        record = self._store.get(DEVICE_PREFIX + did) or {}
        record.update(payload)
        record["device_id"] = did
        record["updated_at"] = time.time()
        # preserve raw device id not stored for privacy
        self._store.set(DEVICE_PREFIX + did, record)
        return record

    def get(self, raw_device_id: str) -> Optional[Dict]:
        did = device_hash(raw_device_id)
        return self._store.get(DEVICE_PREFIX + did)

    def query_capability(self, min_android: int = 0, arch: Optional[str] = None) -> List[Dict]:
        """Return devices that match capability filters (useful for rollout targeting)."""
        results = []
        for key, rec in self._store.scan(DEVICE_PREFIX).items():
            v = rec
            try:
                if int(v.get("android_version", 0)) < min_android:
                    continue
            except Exception:
                pass
            if arch and v.get("arch") != arch:
                continue
            results.append(v)
        return results

    def prune_stale(self, older_than_days: int = CONFIG.device_prune_days) -> int:
        """Remove devices not updated for `older_than_days`. Returns count deleted."""
        cutoff = time.time() - older_than_days * 86400
        deleted = 0
        for key, rec in list(self._store.scan(DEVICE_PREFIX).items()):
            if rec.get("updated_at", 0) < cutoff:
                self._store.delete(key)
                deleted += 1
        return deleted
