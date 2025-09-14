# Tests for query_engine
# monitoring/logging/centralized/tests/test_query_engine.py
import re
from datetime import datetime
from monitoring.logging.centralized.storage import InMemoryStorage
from monitoring.logging.centralized.indexer import LogIndexer
from monitoring.logging.centralized.query_engine import QueryEngine

def test_query_engine():
    storage = InMemoryStorage()
    indexer = LogIndexer()
    log = {"id": "1", "timestamp": datetime.utcnow(), "message": "critical issue"}
    storage.store([log])
    indexer.index([log])
    engine = QueryEngine(storage, indexer)
    results = engine.query(text="critical")
    assert len(results) == 1
