"""
Chunk manager for resumable uploads.

Strategy:
- Each upload gets an upload_id (UUID)
- Chunks are uploaded (index-based) to CHUNK_DIR/{upload_id}/{index}.part
- We maintain a small JSON "manifest" per upload listing total_chunks / received
- When all chunks present, they are concatenated atomically into STAGING_DIR/{upload_id}/{filename}
- Simple concurrency safe design using file existence checks and atomic move
"""

import os
import uuid
import json
from typing import Dict, List
from pathlib import Path

from .config import config
from .utils import ensure_dir, atomic_move, safe_remove, file_hash
from .models import ChunkCompleteEvent, UploadCompleteEvent

ensure_dir(config.CHUNK_DIR)
ensure_dir(config.STAGING_DIR)


def make_upload_id() -> str:
    return uuid.uuid4().hex


def upload_dir(upload_id: str) -> str:
    return os.path.join(config.CHUNK_DIR, upload_id)


def manifest_path(upload_id: str) -> str:
    return os.path.join(upload_dir(upload_id), "manifest.json")


def init_upload(upload_id: str, filename: str, total_bytes: int, chunk_size: int):
    d = upload_dir(upload_id)
    ensure_dir(d)
    manifest = {
        "upload_id": upload_id,
        "filename": filename,
        "total_bytes": total_bytes,
        "chunk_size": chunk_size,
        "chunks": {},  # index -> bytes
    }
    with open(manifest_path(upload_id), "w") as f:
        json.dump(manifest, f)


def write_chunk(upload_id: str, chunk_index: int, data: bytes) -> int:
    d = upload_dir(upload_id)
    ensure_dir(d)
    part_path = os.path.join(d, f"{chunk_index}.part")
    # write chunk atomically using a tmp file and move
    tmp = part_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, part_path)
    # update manifest
    mpath = manifest_path(upload_id)
    with open(mpath, "r+") as f:
        manifest = json.load(f)
        manifest["chunks"][str(chunk_index)] = len(data)
        f.seek(0)
        json.dump(manifest, f)
        f.truncate()
    return len(data)


def list_received_chunks(upload_id: str) -> List[int]:
    mpath = manifest_path(upload_id)
    if not os.path.exists(mpath):
        return []
    with open(mpath, "r") as f:
        manifest = json.load(f)
    return sorted(int(k) for k in manifest.get("chunks", {}).keys())


def is_complete(upload_id: str) -> bool:
    mpath = manifest_path(upload_id)
    if not os.path.exists(mpath):
        return False
    with open(mpath, "r") as f:
        manifest = json.load(f)
    total_bytes = manifest["total_bytes"]
    received = sum(manifest["chunks"].values())
    return received >= total_bytes


def assemble(upload_id: str) -> str:
    """
    Concatenate parts into staged file; return staged file path.
    """
    mpath = manifest_path(upload_id)
    if not os.path.exists(mpath):
        raise FileNotFoundError("manifest not found")
    with open(mpath, "r") as f:
        manifest = json.load(f)
    filename = manifest["filename"]
    staged_dir = os.path.join(config.STAGING_DIR, upload_id)
    ensure_dir(staged_dir)
    staged_path = os.path.join(staged_dir, filename)
    part_indices = sorted(int(k) for k in manifest["chunks"].keys())
    # write to a tmp file then move into place
    tmp_out = staged_path + ".tmp"
    with open(tmp_out, "wb") as outfile:
        for idx in part_indices:
            part_path = os.path.join(upload_dir(upload_id), f"{idx}.part")
            if not os.path.exists(part_path):
                raise FileNotFoundError(f"missing part {idx}")
            with open(part_path, "rb") as pf:
                while True:
                    buf = pf.read(8192)
                    if not buf:
                        break
                    outfile.write(buf)
    os.replace(tmp_out, staged_path)
    return staged_path


def cleanup_upload(upload_id: str):
    """Remove chunk dir and staged files (if any)."""
    d = upload_dir(upload_id)
    staged_d = os.path.join(config.STAGING_DIR, upload_id)
    for p in [d, staged_d]:
        if os.path.exists(p):
            # remove directory tree
            for root, dirs, files in os.walk(p, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except FileNotFoundError:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass
            try:
                os.rmdir(p)
            except Exception:
                pass
