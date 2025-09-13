# High-level registry API
# registry.py
"""
High-level Registry API that ties together storage + metadata.
Provides:
 - register_model: upload artifact and store metadata
 - fetch_model: download artifact to local path
 - list_versions, latest, promote, delete
 - auto-version strategy: semver bump or timestamp tags
"""

from typing import Optional, Dict, Any
from metadata_store import MetadataStore, SQLiteMetadataStore
from storage_adapter import StorageAdapter, LocalStorageAdapter
from model_artifact import ModelArtifact
from utils import compute_sha256, is_semver, bump_semver, now_iso
import os
import shutil
import logging

logger = logging.getLogger("model.registry")


class ModelRegistry:
    def __init__(self, store: Optional[MetadataStore] = None, storage: Optional[StorageAdapter] = None, default_storage_prefix: str = ""):
        self.store = store or SQLiteMetadataStore()
        self.storage = storage or LocalStorageAdapter()
        self.prefix = default_storage_prefix.rstrip("/")

    def _dest_path(self, name: str, version: str, filename: str):
        # destination path within storage root; keep organized by name/version/filename
        base = f"{name}/{version}/{filename}"
        if self.prefix:
            return f"{self.prefix}/{base}"
        return base

    def register_model(self, name: str, src_path: str, version: Optional[str] = None, created_by: Optional[str] = None, metadata: Optional[Dict[str,Any]] = None, provenance: Optional[Dict[str,Any]] = None, auto_bump: str = "patch") -> ModelArtifact:
        """
        Register a model artifact:
          - compute checksum
          - determine version (if not provided, use latest semver bump or timestamp)
          - upload artifact to storage
          - store metadata in metadata store
        """
        if version is None:
            # determine latest and bump semver if possible
            latest = self.store.latest(name)
            if latest and is_semver(latest.version):
                version = bump_semver(latest.version, part=auto_bump)
            else:
                # fallback to timestamp-based version
                version = now_iso().replace(":", "-")
        filename = os.path.basename(src_path)
        dest = self._dest_path(name, version, filename)
        uri = self.storage.upload(src_path, dest)
        sig = compute_sha256(src_path)
        artifact = ModelArtifact(name=name, version=version, uri=uri, created_by=created_by, metadata=metadata or {}, provenance=provenance or {}, signature=sig)
        self.store.register(artifact)
        logger.info(f"Registered model {name}:{version} -> {uri}")
        return artifact

    def fetch_model(self, name: str, version: str, target_path: str):
        art = self.store.get(name, version)
        if not art:
            raise KeyError("Model version not found")
        # attempt to download and return local path
        return self.storage.download(art.uri, target_path)

    def list_versions(self, name: str):
        return self.store.list_versions(name)

    def latest(self, name: str):
        return self.store.latest(name)

    def promote(self, name: str, version: str, tag: str = "staging"):
        self.store.promote(name, version, tag)

    def delete(self, name: str, version: str):
        art = self.store.get(name, version)
        if not art:
            raise KeyError("Model version not found")
        if hasattr(self.storage, "delete"):
            try:
                self.storage.delete(art.uri)
            except Exception:
                logger.warning("Storage adapter failed to delete artifact; proceeding to metadata delete")
        self.store.delete(name, version)
