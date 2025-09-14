# Tests for indexer
# monitoring/logging/centralized/tests/test_indexer.py
from monitoring.logging.centralized.indexer import LogIndexer

def test_indexer_search():
    indexer = LogIndexer()
    logs = [{"id": "1", "message": "error in service"}]
    indexer.index(logs)
    results = indexer.search_full_text("error")
    assert len(results) == 1
