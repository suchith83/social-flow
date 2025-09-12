# utils.py
import hashlib
import logging
from datetime import datetime

logging.basicConfig(
    filename="financial_audit.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def secure_hash(data: str) -> str:
    """Generate SHA-256 hash for sensitive data."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def log_event(message: str, level="INFO"):
    """Centralized logging for compliance events."""
    if level == "ERROR":
        logging.error(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ALERT":
        logging.critical(message)
    else:
        logging.info(message)
    print(f"[{level}] {message}")  # Console output for visibility
