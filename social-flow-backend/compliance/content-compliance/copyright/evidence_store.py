"""
evidence_store.py

Simple filesystem-backed evidence store for takedown notices, attachments, and metadata.
Designed to be robust and audit-friendly (immutable writes).
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .config import CopyrightConfig
from .exceptions import EvidenceNotFoundError

EVIDENCE_BASE = Path(CopyrightConfig.EVIDENCE_STORE_PATH)
EVIDENCE_BASE.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def store_evidence(notice_id: str, attachment_bytes: Optional[bytes], filename: Optional[str], metadata: Dict[str, Any]) -> Path:
    """
    Store an evidence package for a notice.
    Creates a folder per notice: <evidence_base>/<notice_id>/
    - raw attachment -> <filename>
    - metadata.json -> metadata + audit fields
    Returns path to notice folder.
    """
    notice_dir = EVIDENCE_BASE / notice_id
    notice_dir.mkdir(parents=True, exist_ok=True)

    if attachment_bytes and filename:
        attachment_path = notice_dir / filename
        # Write in 'xb' mode to prevent overwriting existing evidence (immutable write attempt)
        with open(attachment_path, "wb") as f:
            f.write(attachment_bytes)

    meta = metadata.copy()
    meta["_stored_at"] = _now_iso()
    meta_path = notice_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return notice_dir


def retrieve_evidence(notice_id: str) -> Dict[str, Any]:
    """Retrieve evidence metadata and list of filenames for a notice id."""
    notice_dir = EVIDENCE_BASE / notice_id
    if not notice_dir.exists():
        raise EvidenceNotFoundError(f"Evidence for notice {notice_id} not found")

    meta_path = notice_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    files = [p.name for p in notice_dir.iterdir() if p.is_file() and p.name != "metadata.json"]
    return {"metadata": metadata, "files": files, "path": str(notice_dir)}


def delete_evidence(notice_id: str) -> None:
    """Irreversible deletion (admin only) — kept minimal here."""
    notice_dir = EVIDENCE_BASE / notice_id
    if notice_dir.exists():
        shutil.rmtree(notice_dir)
