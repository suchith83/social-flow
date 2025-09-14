# Tests for collector
# monitoring/logging/centralized/tests/test_collector.py
import pytest
from monitoring.logging.centralized.collector import LogCollector
from monitoring.logging.centralized.storage import InMemoryStorage
from monitoring.logging.centralized.indexer import LogIndexer

def test_collector_flush():
    storage = InMemoryStorage()
    indexer = LogIndexer()
    collector = LogCollector(storage, indexer, batch_size=2)
    collector.collect({"timestamp": None, "message": "hello", "level": "info"})
    collector.collect({"timestamp": None, "message": "world", "level": "warn"})
    assert len(storage.query_all()) == 2
