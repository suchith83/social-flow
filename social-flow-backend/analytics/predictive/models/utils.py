import os
import json
import joblib
import logging
from datetime import datetime
from typing import Any

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("predictive-models")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)
    logger.info(f"Saved object to {path}")


def load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = joblib.load(path)
    logger.info(f"Loaded object from {path}")
    return obj


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def to_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=str, ensure_ascii=False, indent=2)
    logger.info(f"Wrote JSON to {path}")
