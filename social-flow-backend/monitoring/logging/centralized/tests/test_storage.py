# Tests for storage
# monitoring/logging/centralized/tests/test_storage.py
from monitoring.logging.centralized.storage import InMemoryStorage

def test_inmemory_storage():
    storage = InMemoryStorage()
    log = {"timestamp": None, "message": "test"}
    storage.store([log])
    all_logs = storage.query_all()
    assert len(all_logs) == 1
