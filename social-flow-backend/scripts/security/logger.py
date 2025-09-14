# scripts/security/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

def configure_logging(log_path: str = "/var/log/socialflow/security.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    # Also log to console for CI
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
