# Config + helpers
# ================================================================
# File: utils.py
# Purpose: Config loading + logger setup
# ================================================================

import yaml
import logging


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
