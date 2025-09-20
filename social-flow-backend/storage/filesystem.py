"""Simple filesystem storage utilities used as a local fallback for object storage."""
from typing import BinaryIO, Optional
import os
import errno

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_STORAGE_DIR = os.environ.get("LOCAL_STORAGE_DIR", os.path.join(ROOT, ".local_storage"))


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # best-effort
        pass


def save_stream(fileobj: BinaryIO, relative_path: str) -> str:
    """Save fileobj to LOCAL_STORAGE_DIR/relative_path. Returns absolute path."""
    dst = os.path.join(LOCAL_STORAGE_DIR, relative_path)
    _ensure_dir(os.path.dirname(dst))
    with open(dst, "wb") as f:
        fileobj.seek(0)
        while True:
            chunk = fileobj.read(8192)
            if not chunk:
                break
            f.write(chunk)
    return os.path.abspath(dst)


def read_stream(relative_path: str, fileobj: BinaryIO) -> None:
    src = os.path.join(LOCAL_STORAGE_DIR, relative_path)
    with open(src, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            fileobj.write(chunk)
    fileobj.seek(0)


def remove(relative_path: str) -> None:
    p = os.path.join(LOCAL_STORAGE_DIR, relative_path)
    try:
        os.remove(p)
    except FileNotFoundError:
        pass
