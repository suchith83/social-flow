# =========================
# File: testing/security/penetration/utils/logger.py
# =========================
"""
Central logger for penetration package. Configured to avoid leaking secrets into logs.
"""

import logging

def get_logger(name="Pentest"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
