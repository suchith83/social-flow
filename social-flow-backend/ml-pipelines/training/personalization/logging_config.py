# Logging configuration for personalization
# ============================
# File: logging_config.py
# ============================
import logging, sys

def setup_logger(name="personalization", level=logging.INFO):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log = logging.getLogger(name); log.setLevel(level); log.addHandler(h)
    return log
