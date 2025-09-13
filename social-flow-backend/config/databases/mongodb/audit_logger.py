"""Audit logging for reads/writes with structured logs."""
"""
audit_logger.py
---------------
Structured audit logging for MongoDB operations (reads/writes).
- Uses a JSON structured logger, stores logs to file, and optionally ships to external logging system.
- Fields: timestamp, user_id (if present), operation, collection, query, doc_id, duration_ms, outcome
"""

import logging
import json
from time import time
from pathlib import Path

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)
AUDIT_LOG_FILE = LOG_PATH / "mongodb_audit.log"

logger = logging.getLogger("MongoDBAudit")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(AUDIT_LOG_FILE)
fh.setFormatter(logging.Formatter("%(message)s"))  # store raw JSON lines
logger.addHandler(fh)


def _emit(record: dict):
    try:
        logger.info(json.dumps(record, default=str))
    except Exception:
        logger.exception("Failed to emit audit record")


def audit_event(user_id: str = None, operation: str = None, collection: str = None,
                query: dict = None, doc_id: str = None, duration_ms: float = None, outcome: str = "ok", metadata: dict = None):
    rec = {
        "timestamp": time(),
        "user_id": user_id,
        "operation": operation,
        "collection": collection,
        "query": query,
        "doc_id": doc_id,
        "duration_ms": duration_ms,
        "outcome": outcome,
        "metadata": metadata or {}
    }
    _emit(rec)


# Example helper context manager to measure operation duration
from contextlib import contextmanager

@contextmanager
def audit_context(user_id=None, operation=None, collection=None, query=None, doc_id=None, metadata=None):
    start = time()
    try:
        yield
        duration = (time() - start) * 1000.0
        audit_event(user_id=user_id, operation=operation, collection=collection,
                    query=query, doc_id=doc_id, duration_ms=duration, outcome="ok", metadata=metadata)
    except Exception as e:
        duration = (time() - start) * 1000.0
        audit_event(user_id=user_id, operation=operation, collection=collection,
                    query=query, doc_id=doc_id, duration_ms=duration, outcome="error", metadata={"error": str(e)})
        raise
