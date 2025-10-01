"""Compliance logging of Elasticsearch queries."""
"""
audit_logger.py
---------------
Logs Elasticsearch queries for compliance & auditing.
"""

import logging

logger = logging.getLogger("ElasticsearchAudit")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/elasticsearch_audit.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


def log_query(user: str, query: dict, index: str):
    """Log query for compliance."""
    logger.info(f"User: {user}, Index: {index}, Query: {query}")
