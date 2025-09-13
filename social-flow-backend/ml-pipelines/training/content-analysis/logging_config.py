# Configure structured logging
# ============================
# File: logging_config.py
# ============================
import logging
import sys

def setup_logger(name="content-analysis", level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
