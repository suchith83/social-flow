"""
Utility functions for Dynamic Analysis
"""

import logging
import json
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("dynamic_analysis.log"), logging.StreamHandler()]
)

logger = logging.getLogger("dynamic-analysis")


def save_json(filepath: str, data: dict):
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def timestamp() -> str:
    """Return UTC timestamp string."""
    return datetime.utcnow().isoformat() + "Z"
