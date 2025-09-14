# Helper functions for dates, compression, size calculations
# monitoring/logging/retention/utils.py
"""
Utility helpers for retention system.
Includes date handling, size calculations, and compression functions.
"""

import os
import gzip
import shutil
import datetime
from pathlib import Path


def days_between(d1, d2) -> int:
    """Return number of days between two datetimes."""
    return abs((d2 - d1).days)


def get_file_age_days(path: Path) -> int:
    """Get file age in days from modification time."""
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return days_between(mtime, datetime.datetime.utcnow())


def compress_file(path: Path, algo="gzip") -> Path:
    """Compress file with selected algorithm."""
    if algo == "gzip":
        out_path = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return out_path
    elif algo == "zip":
        import zipfile
        out_path = path.with_suffix(".zip")
        with zipfile.ZipFile(out_path, "w") as z:
            z.write(path, arcname=path.name)
        return out_path
    return path


def folder_size(path: Path) -> int:
    """Calculate total folder size in bytes."""
    return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
