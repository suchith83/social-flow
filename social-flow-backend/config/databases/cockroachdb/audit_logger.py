"""Ensures compliance logging for all DB transactions."""
"""
audit_logger.py
---------------
Logs all database transactions for compliance (GDPR, CCPA, LGPD).
"""

import logging

logger = logging.getLogger("AuditLogger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/cockroach_audit.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


def log_query(user: str, query: str):
    """Log SQL query with user info."""
    logger.info(f"User: {user}, Query: {query}")
