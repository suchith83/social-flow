"""
Small helpers to handle test file paths, downloads, and temporary directories.
"""

from pathlib import Path
import tempfile
import shutil


def fixtures_dir():
    return Path(__file__).resolve().parents[1] / "fixtures"


def sample_file_path(name: str):
    return fixtures_dir() / name


def temp_dir():
    return Path(tempfile.mkdtemp())


def cleanup_dir(path: Path):
    shutil.rmtree(path, ignore_errors=True)
