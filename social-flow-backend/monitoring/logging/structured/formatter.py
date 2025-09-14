# Structured logging formatter (plugs into Python logging)
"""
A logging.Formatter subclass that emits structured logs.

Usage:
    handler = logging.FileHandler("structured.ndjson")
    handler.setFormatter(StructuredJSONFormatter(service="my-service"))
    logger.addHandler(handler)
"""

import logging
import socket
from typing import Optional
from .serializer import to_ndjson_line
from .transformer import enrich
from .schema import StructuredLog
from datetime import datetime


class StructuredJSONFormatter(logging.Formatter):
    def __init__(self, service: Optional[str] = None, include_host: bool = True):
        super().__init__()
        self.service = service
        self.host = socket.gethostname() if include_host else None

    def format(self, record: logging.LogRecord) -> str:
        # build structured dict
        payload = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "service": self.service,
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "host": self.host,
            "attrs": {}
        }

        # attach extra fields from record.__dict__ (but avoid dupes and internals)
        for k, v in getattr(record, "__dict__", {}).items():
            if k in {"message", "msg", "args", "levelno", "levelname", "name", "created"}:
                continue
            if k.startswith("_"):
                continue
            payload.setdefault("attrs", {})[k] = v

        # enrich and validate via pydantic (defensive)
        payload = enrich(payload)
        try:
            StructuredLog.parse_obj(payload)
        except Exception:
            # swallow schema errors: fallback to a minimal payload
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "service": self.service or "unknown",
                "level": record.levelname,
                "message": record.getMessage()
            }
        return to_ndjson_line(payload)
