"""Advanced search queries: full-text, filters, aggregations."""
"""
search_manager.py
-----------------
Handles advanced search queries: full-text, filters, and aggregations.
"""

from .connection import ElasticsearchClient
import logging

logger = logging.getLogger("SearchManager")
logger.setLevel(logging.INFO)


class SearchManager:
    def __init__(self):
        self.client = ElasticsearchClient().get_client()

    def full_text_search(self, index: str, field: str, query: str):
        """Perform full-text search."""
        body = {"query": {"match": {field: query}}}
        return self.client.search(index=index, body=body)

    def filter_search(self, index: str, filters: dict):
        """Perform search with filters."""
        body = {"query": {"bool": {"filter": [{"term": {k: v}} for k, v in filters.items()]}}}
        return self.client.search(index=index, body=body)

    def aggregation_search(self, index: str, field: str):
        """Run aggregation query."""
        body = {"aggs": {"value_counts": {"terms": {"field": field}}}}
        return self.client.search(index=index, body=body, size=0)
