# Unit tests for cache storage
"""
Unit tests for storage layer (DiskStore mostly).
"""

import os
import tempfile
from mobile_backend.offline.content_caching.storage import DiskStore
import pytest

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)

def test_disk_write_and_read(tmp_dir):
    ds = DiskStore(tmp_dir)
    key = "content:1"
    data = b"hello world"
    p, size = ds.write(key, data)
    assert size == len(data)
    assert ds.exists(key)
    read = ds.read(key)
    assert read == data
    ds.delete(key)
    assert not ds.exists(key)
