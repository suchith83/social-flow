# Indexing & retrieval of logs
# monitoring/logging/centralized/indexer.py
"""
Indexing engine for centralized logging.
Supports full-text search and structured filters.
"""

import re
from collections import defaultdict
from .config import CONFIG


class LogIndexer:
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.logs = {}

    def index(self, logs: list):
        """Index logs for fast retrieval."""
        for log in logs:
            log_id = log.get("id")
            self.logs[log_id] = log
            if CONFIG["INDEXER"]["enable_full_text"]:
                for word in str(log.get("message", "")).split():
                    self.inverted_index[word.lower()].add(log_id)

    def search_full_text(self, term: str):
        """Full-text search logs by term."""
        ids = self.inverted_index.get(term.lower(), set())
        return [self.logs[i] for i in ids]

    def filter_by(self, field: str, pattern: str):
        """Regex filter logs by field."""
        results = []
        for log in self.logs.values():
            if re.search(pattern, str(log.get(field, ""))):
                results.append(log)
        return results
