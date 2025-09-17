import pytest
from storage.object-storage.multi-cloud.factory import StorageFactory


def test_invalid_provider(monkeypatch):
    monkeypatch.setenv("CLOUD_PROVIDER", "invalid")
    with pytest.raises(ValueError):
        StorageFactory.get_storage()
