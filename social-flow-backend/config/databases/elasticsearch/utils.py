"""Helper utilities for debugging and schema validation."""
"""
utils.py
--------
Helper utilities for Elasticsearch diagnostics and debugging.
"""

from .connection import ElasticsearchClient


def list_indices():
    """List all indices."""
    client = ElasticsearchClient().get_client()
    return list(client.indices.get_alias("*").keys())


def check_index_mapping(index: str):
    """Get mapping of an index."""
    client = ElasticsearchClient().get_client()
    return client.indices.get_mapping(index=index)
