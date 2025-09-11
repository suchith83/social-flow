# logger.py
import logging
import json
import sys
import uuid
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom JSON log formatter for structured logging.
    Adds timestamp, log level, correlation_id and message in JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", None),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def get_logger(name: str = "monitoring", level=logging.INFO, correlation_id=None) -> logging.Logger:
    """
    Returns a logger instance with JSON formatting.
    Supports correlation IDs for distributed tracing.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)

    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    logger = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    return logger
