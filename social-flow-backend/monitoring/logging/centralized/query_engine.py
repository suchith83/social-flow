# Advanced query API (filters, regex, time range, full-text)
# monitoring/logging/centralized/query_engine.py
"""
Query engine for centralized logging.
Provides advanced filters, full-text, and time range queries.
"""

from datetime import datetime


class QueryEngine:
    def __init__(self, storage, indexer):
        self.storage = storage
        self.indexer = indexer

    def query(self, text=None, field=None, regex=None,
              start_time=None, end_time=None):
        """Query logs with multiple filters."""
        logs = self.storage.query_all()

        if text:
            logs = self.indexer.search_full_text(text)

        if field and regex:
            logs = [log for log in logs if regex.search(str(log.get(field, "")))]

        if start_time:
            logs = [log for log in logs if log.get("timestamp") and log["timestamp"] >= start_time]
        if end_time:
            logs = [log for log in logs if log.get("timestamp") and log["timestamp"] <= end_time]

        return logs
