# Unit tests for cache routes
"""
Integration-style tests for routes using TestClient and SQLite disk store.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from mobile_backend.offline.content_caching.routes import router
from mobile_backend.offline.content_caching.database import init_db
from mobile_backend.offline.content_caching.storage import DiskStore
import os

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_db_and_disk(tmp_path_factory):
    init_db()
    # override disk path to tmp for tests
    from mobile_backend.offline.content_caching.config import get_config
    cfg = get_config()
    cfg.DISK_CACHE_PATH = str(tmp_path_factory.mktemp("cache"))
    yield

def test_put_and_get_cache():
    files = {"file": ("blob.bin", b"abc123")}
    key = "content:test:1"
    r = client.post(f"/offline/cache/?key={key}", files=files)
    assert r.status_code == 201
    # fetch
    g = client.get(f"/offline/cache/{key}")
    assert g.status_code == 200
    assert g.content == b"abc123"

def test_head_cache_miss():
    r = client.head("/offline/cache/nonexistent")
    assert r.status_code == 404
