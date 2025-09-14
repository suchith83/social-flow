# Handles compression and archive migration
# monitoring/logging/retention/archiver.py
"""
Archiver for retention system.
Handles compression and migration of old logs into archive storage.
"""

from pathlib import Path
from .config import CONFIG
from .utils import compress_file


class Archiver:
    def __init__(self):
        self.archive_path = CONFIG["ARCHIVE"]["path"]
        self.archive_path.mkdir(parents=True, exist_ok=True)
        self.algo = CONFIG["ARCHIVE"]["compression"]

    def archive(self, file_path: Path) -> Path:
        """Archive a file (compress and move)."""
        compressed = compress_file(file_path, self.algo)
        dest = self.archive_path / compressed.name
        compressed.replace(dest)
        return dest
