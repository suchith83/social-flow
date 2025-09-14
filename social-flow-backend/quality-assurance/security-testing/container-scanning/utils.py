"""
Utility functions for container scanning.
"""

import hashlib
import json
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("scanner.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("container-scanning")


def hash_image(image_name: str) -> str:
    """Generate a deterministic hash for an image name."""
    return hashlib.sha256(image_name.encode()).hexdigest()


def save_json(filepath: str, data: dict):
    """Save dictionary as JSON file with safe handling."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def timestamp() -> str:
    """Return formatted timestamp."""
    return datetime.utcnow().isoformat() + "Z"
