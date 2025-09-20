# logger.py
import logging
import os
import json
import sys
import uuid
from datetime import datetime
from typing import Optional

LOG_JSON = os.environ.get("SF_LOG_JSON", "false").lower() in ("1", "true", "yes")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str, level: Optional[int] = logging.INFO, correlation_id=None) -> logging.Logger:
    """
    Return a configured logger. Safe to call multiple times.
    Uses simple stream handler and optional JSON formatting controlled by SF_LOG_JSON.
    """
    l = logging.getLogger(name)
    if not l.handlers:
        h = logging.StreamHandler(sys.stdout)
        if LOG_JSON:
            h.setFormatter(JsonFormatter())
        else:
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            h.setFormatter(fmt)
        l.addHandler(h)
    l.setLevel(level)

    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    l = logging.LoggerAdapter(l, {"correlation_id": correlation_id})
    return l
