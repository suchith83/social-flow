import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture(scope="session")
def fake_db():
    """
    Provides a fake in-memory database (dictionary-based).
    Used for fast isolated testing without real DB connections.
    """
    return {"users": {}, "videos": {}, "payments": {}}

@pytest.fixture
def mock_cache():
    """
    Simulates a Redis-like cache using a simple dictionary.
    """
    cache = {}
    yield cache
    cache.clear()

@pytest.fixture
def mock_s3_client():
    """
    Provides a fake AWS S3 client with stubbed upload/download.
    """
    client = MagicMock()
    client.upload_file.return_value = True
    client.download_file.return_value = b"fake-binary-content"
    return client

@pytest.fixture
def async_mock_service():
    """
    Creates an async mock service, useful for async/await flows.
    """
    return AsyncMock()
