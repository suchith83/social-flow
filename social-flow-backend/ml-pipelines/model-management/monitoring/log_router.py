# Centralized log routing and filtering
"""
log_router.py
Very small helper to route application logs to different sinks:
 - stdout (console)
 - file
 - optionally to ELK / Loki / HTTP endpoint (simple POST)
Provides structured JSON logs out of the box.
"""

import logging
import json
import sys
from logging import Logger
from typing import Optional
import requests
from utils import setup_logger

# default logger wrapper
class JSONFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logger(name: str = "ml-monitor", level=logging.INFO, out_file: Optional[str] = None, push_url: Optional[str] = None) -> Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = JSONFormatter()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if out_file:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if push_url:
        # attach a simple handler that posts logs (be careful for perf)
        class HTTPHandler(logging.Handler):
            def emit(self, record):
                try:
                    payload = {"log": self.format(record)}
                    requests.post(push_url, json=payload, timeout=1)
                except Exception:
                    pass
        h = HTTPHandler()
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
