"""Helper utilities: connection diagnostics, safe reads, fan-out helpers, ID utilities."""
"""
utils.py
--------
Misc helpers for MongoDB:
 - connection diagnostics
 - safe read helpers with retries
 - id helpers for conversion between str and ObjectId
 - simple fan-out helper for bulk ops
"""

from typing import Iterable, Any, Callable
import time
from bson import ObjectId
from pymongo.errors import PyMongoError
from .connection import MongoConnectionManager
import logging

logger = logging.getLogger("MongoUtils")
logger.setLevel(logging.INFO)


def to_object_id(value: Any) -> ObjectId:
    """Convert string or ObjectId-friendly value into ObjectId, raising on invalid."""
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception as e:
        raise ValueError(f"Invalid ObjectId value: {value}") from e


def safe_find_one(collection, filter_doc, max_retries: int = 3, backoff: float = 0.5):
    """Safe read with simple retry logic for transient network errors."""
    attempt = 0
    while True:
        try:
            return collection.find_one(filter_doc)
        except PyMongoError as e:
            attempt += 1
            if attempt > max_retries:
                logger.exception("safe_find_one exhausted retries")
                raise
            logger.warning(f"safe_find_one transient error, retry {attempt}/{max_retries}: {e}")
            time.sleep(backoff * (2 ** (attempt - 1)))


def bulk_process(iterable: Iterable, fn: Callable, batch: int = 500):
    """
    Apply `fn` to items in batches. `fn` receives a list of items.
    Example: bulk_process(cursor, lambda batch: collection.insert_many(batch), batch=1000)
    """
    buffer = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) >= batch:
            fn(buffer)
            buffer = []
    if buffer:
        fn(buffer)
