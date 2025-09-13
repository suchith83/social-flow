# Device records and capability detection
"""
Device registry tuned for Flutter clients.

Tracks:
 - Flutter engine version
 - Flutter framework version
 - Dart VM / AOT mode or JIT (development)
 - Supported ABIs and target platforms (android-arm64, ios-arm64, etc.)
 - Installed module versions (for incremental updates)
 - Push tokens for each platform (fcm token for Android, apns token for iOS)

This module uses the InMemoryStore from utils for demo; replace with DB in prod.
"""

import time
from typing import Dict, Optional, List
from .utils import store, device_hash, ensure_dir, file_size
from .config import CONFIG

DEVICE_PREFIX = "flutter_device:"


class FlutterDeviceRegistry:
    def __init__(self):
        self._store = store

    def register(self, raw_device_id: str, payload: Dict) -> Dict:
        """
        Register or update a device.
        payload example:
         {
           "platform": "android"|"ios",
           "abi": "arm64-v8a"|"arm64",
           "flutter_version": "3.3.2",
           "engine_version": "2.14.0",
           "dart_version": "2.18.0",
           "app_version": "1.2.3",
           "push_token": "...",
           "installed_bundle_sha": "..."
         }
        """
        did = device_hash(raw_device_id, salt="flutter-salt")
        key = DEVICE_PREFIX + did
        record = self._store.get(key) or {}
        record.update(payload)
        record["device_id"] = did
        record["updated_at"] = time.time()
        self._store.set(key, record)
        return record

    def get(self, raw_device_id: str) -> Optional[Dict]:
        did = device_hash(raw_device_id, salt="flutter-salt")
        return self._store.get(DEVICE_PREFIX + did)

    def query_by_capability(self, min_flutter: Optional[str] = None, platform: Optional[str] = None, abi: Optional[str] = None) -> List[Dict]:
        """Return devices matching capability filters for targeted rollouts."""
        out = []
        for k, v in self._store.scan(DEVICE_PREFIX).items():
            if platform and v.get("platform") != platform:
                continue
            if abi and v.get("abi") != abi:
                continue
            if min_flutter:
                try:
                    if tuple(map(int, (v.get("flutter_version") or "0.0.0").split("."))) < tuple(map(int, min_flutter.split("."))):
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
