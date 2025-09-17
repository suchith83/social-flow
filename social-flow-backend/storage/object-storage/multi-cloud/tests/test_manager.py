import pytest
from storage.object-storage.multi-cloud.manager import MultiCloudStorageManager


def test_manager_methods(monkeypatch):
    # Fake provider
    class DummyBackend:
        def upload_file(self, f, k): return "uploaded"
        def download_file(self, k, f): return "downloaded"
        async def async_upload_file(self, f, k): return "async_uploaded"
        async def async_download_file(self, k, f): return "async_downloaded"
        def generate_presigned_url(self, k, e): return "url"

    monkeypatch.setattr("storage.object-storage.multi-cloud.factory.StorageFactory.get_storage", lambda: DummyBackend())

    mgr = MultiCloudStorageManager()
    assert mgr.upload_file("a", "b") == "uploaded"
    assert mgr.download_file("a", "b") == "downloaded"
    assert mgr.generate_presigned_url("a") == "url"
