# Implements different storage backends (S3, local FS, DB)
"""
Simple pluggable storage backends used by TieredStorage.

These are lightweight, easily-replaceable adapters for actual production backends
(like S3, GCS, Azure Blob, Ceph, or a block storage service). This module includes:
 - LocalDiskBackend: used for hot tier (fast local access) — uses files on disk.
 - S3Backend: sketch of a remote object store interface (no boto3 dependency).
 - ColdArchiveBackend: placeholder for tape/archive-like backend (simulated).

In a real system these would wrap cloud SDKs and include retry policies, multipart uploads,
encryption integration and server-side lifecycle configuration.
"""

import os
import hashlib
import threading
from typing import Tuple, Optional

class BaseBackend:
    def put(self, key: str, data: bytes, metadata: dict = None) -> None:
        raise NotImplementedError

    def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

class LocalDiskBackend(BaseBackend):
    """
    Local filesystem backend for hot tier. Thread-safe file operations with simple layout.
    """
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.lock = threading.RLock()

    def _path(self, key: str) -> str:
        # Sanitize key minimally; production should be stricter
        safe_key = key.replace("/", "_")
        return os.path.join(self.root_dir, safe_key)

    def put(self, key: str, data: bytes, metadata: dict = None) -> None:
        path = self._path(key)
        with self.lock:
            with open(path, "wb") as f:
                f.write(data)
            if metadata:
                meta_path = path + ".meta"
                with open(meta_path, "w", encoding="utf-8") as m:
                    m.write(str(metadata))

    def get(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        with self.lock:
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                return f.read()

    def delete(self, key: str) -> None:
        path = self._path(key)
        with self.lock:
            try:
                if os.path.exists(path):
                    os.remove(path)
                meta = path + ".meta"
                if os.path.exists(meta):
                    os.remove(meta)
            except OSError:
                pass

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

class S3Backend(BaseBackend):
    """
    Sketch of remote backend. In prod, replace body with boto3 calls with retries,
    multipart upload, encryption, etc.
    """
    def __init__(self, bucket_name: str, prefix: str = ""):
        self.bucket = bucket_name
        self.prefix = prefix.rstrip("/")

        # For the purpose of the demo, use a local directory mapping to simulate remote store
        root = f".mock_s3_{self.bucket}"
        os.makedirs(root, exist_ok=True)
        self.sim_root = root

    def _path(self, key: str) -> str:
        safe_key = key.replace("/", "_")
        return os.path.join(self.sim_root, safe_key)

    def put(self, key: str, data: bytes, metadata: dict = None) -> None:
        # Simulate network latency & store
        path = self._path(key)
        with open(path, "wb") as f:
            f.write(data)

    def get(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

class ColdArchiveBackend(BaseBackend):
    """
    Simulates a cold archive backend: very slow retrieval, long retention, cheaper storage.
    For production -- would use Glacier, Deep Archive, or custom tape system with retrieval jobs.
    """
    def __init__(self, archive_dir: str):
        self.archive_dir = os.path.abspath(archive_dir)
        os.makedirs(self.archive_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe_key = key.replace("/", "_")
        return os.path.join(self.archive_dir, safe_key)

    def put(self, key: str, data: bytes, metadata: dict = None) -> None:
        # Simulate asynchronous ingest: write but mark as slow retrieval
        path = self._path(key)
        with open(path, "wb") as f:
            f.write(data)

    def get(self, key: str) -> Optional[bytes]:
        # Simulate that cold retrieval is allowed but expensive
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))
