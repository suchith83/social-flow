# SQLite metadata store (pluggable interface)
# metadata_store.py
"""
Pluggable metadata store interface + default SQLite implementation.
Stores model metadata JSON and indexes for quick queries.
This is intentionally simple (no migrations) and designed to be swapped
for Postgres or an external model registry service later.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
from model_artifact import ModelArtifact
import json
from utils import now_iso
import threading


SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    uri TEXT NOT NULL,
    metadata TEXT,
    provenance TEXT,
    signature TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    UNIQUE(name, version)
);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_name_version ON models(name, version);
"""


class MetadataStore:
    """
    Simple metadata store interface.
    """

    def register(self, artifact: ModelArtifact) -> None:
        raise NotImplementedError

    def list_versions(self, name: str) -> List[ModelArtifact]:
        raise NotImplementedError

    def get(self, name: str, version: str) -> Optional[ModelArtifact]:
        raise NotImplementedError

    def latest(self, name: str) -> Optional[ModelArtifact]:
        raise NotImplementedError

    def promote(self, name: str, version: str, tag: str) -> None:
        raise NotImplementedError

    def delete(self, name: str, version: str) -> None:
        raise NotImplementedError


class SQLiteMetadataStore(MetadataStore):
    """
    SQLite metadata store implementation.
    Thread-safe via a lock.
    """

    def __init__(self, db_path: str = "model_registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(SCHEMA)
            self._conn.commit()

    def register(self, artifact: ModelArtifact) -> None:
        payload = json.dumps(artifact.metadata)
        prov = json.dumps(artifact.provenance)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO models (name, version, uri, metadata, provenance, signature, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact.name, artifact.version, artifact.uri, payload, prov, artifact.signature, now_iso(), artifact.created_by),
            )
            self._conn.commit()

    def list_versions(self, name: str):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT name, version, uri, metadata, provenance, signature, created_at, created_by FROM models WHERE name = ? ORDER BY created_at DESC", (name,))
            rows = cur.fetchall()
        artifacts = []
        for r in rows:
            metadata = json.loads(r[3]) if r[3] else {}
            prov = json.loads(r[4]) if r[4] else {}
            artifacts.append(ModelArtifact(name=r[0], version=r[1], uri=r[2], created_by=r[7], metadata=metadata, provenance=prov, signature=r[5]))
        return artifacts

    def get(self, name: str, version: str):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT name, version, uri, metadata, provenance, signature, created_at, created_by FROM models WHERE name = ? AND version = ?", (name, version))
            r = cur.fetchone()
        if not r:
            return None
        metadata = json.loads(r[3]) if r[3] else {}
        prov = json.loads(r[4]) if r[4] else {}
        return ModelArtifact(name=r[0], version=r[1], uri=r[2], created_by=r[7], metadata=metadata, provenance=prov, signature=r[5])

    def latest(self, name: str):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT name, version, uri, metadata, provenance, signature, created_at, created_by FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1", (name,))
            r = cur.fetchone()
        if not r:
            return None
        metadata = json.loads(r[3]) if r[3] else {}
        prov = json.loads(r[4]) if r[4] else {}
        return ModelArtifact(name=r[0], version=r[1], uri=r[2], created_by=r[7], metadata=metadata, provenance=prov, signature=r[5])

    def promote(self, name: str, version: str, tag: str):
        """
        Implement 'promotion' by creating a duplicate row with a specialized version string,
        e.g., '1.2.3 -> promoted:staging' or set of tags (this simple store duplicates).
        In more advanced stores, you'd have tags table.
        """
        existing = self.get(name, version)
        if not existing:
            raise KeyError("Version not found")
        promoted_version = f"{existing.version}+{tag}"
        promoted = ModelArtifact(name=existing.name, version=promoted_version, uri=existing.uri, created_by="promotion", metadata=existing.metadata, provenance=existing.provenance, signature=existing.signature)
        self.register(promoted)

    def delete(self, name: str, version: str):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM models WHERE name = ? AND version = ?", (name, version))
            self._conn.commit()
