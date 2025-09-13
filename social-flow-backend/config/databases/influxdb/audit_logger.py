"""Compliance logging of all queries and writes."""
"""
audit_logger.py
---------------
Logs queries and writes for compliance & auditing.
"""

import logging

logger = logging.getLogger("InfluxAudit")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/influxdb_audit.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


def log_query(user: str, query: str, query_type: str = "Flux"):
    """Log query execution."""
    logger.info(f"User: {user}, Type: {query_type}, Query: {query}")


def log_write(user: str, measurement: str, fields: dict, tags: dict):
    """Log write operation."""
    logger.info(f"User: {user}, Measurement: {measurement}, Fields: {fields}, Tags: {tags}")
