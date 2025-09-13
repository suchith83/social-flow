# Helpers (hashing, timestamps, semver)
# utils.py
"""
Utility helpers for the versioning module.
 - semver helpers (increment patch/minor/major)
 - hashing helpers for artifacts
 - timestamp formatting
"""

import hashlib
import time
from typing import Tuple
import re


SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$")


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts()))


def compute_sha256(path_or_bytes) -> str:
    """
    Compute SHA256 of a file path OR bytes buffer.
    """
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
        return h.hexdigest()
    # assume path string
    with open(path_or_bytes, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_semver(v: str) -> Tuple[int, int, int, str]:
    m = SEMVER_RE.match(v)
    if not m:
        raise ValueError(f"Invalid semver: {v}")
    major, minor, patch, extra = m.groups()
    return int(major), int(minor), int(patch), extra or ""


def bump_semver(v: str, part: str = "patch") -> str:
    major, minor, patch, extra = parse_semver(v)
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError("part must be 'major'|'minor'|'patch'")
    base = f"{major}.{minor}.{patch}"
    if extra:
        return f"{base}-{extra}"
    return base


def is_semver(v: str) -> bool:
    return bool(SEMVER_RE.match(v))
