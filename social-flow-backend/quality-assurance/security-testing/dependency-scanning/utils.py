"""
Utility functions for dependency scanning.
"""

import hashlib
import json
import logging
import os
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("dependency_scanner.log"), logging.StreamHandler()]
)

logger = logging.getLogger("dependency-scanning")


def hash_dependency(dep_name: str, version: str) -> str:
    """Hash dependency name+version for consistent keying."""
    return hashlib.sha256(f"{dep_name}:{version}".encode()).hexdigest()


def save_json(filepath: str, data: dict):
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def timestamp() -> str:
    """Return UTC timestamp string."""
    return datetime.utcnow().isoformat() + "Z"
