import pytest
from storage.object-storage.google-cloud-storage.downloader import GCSDownloader


def test_downloader_initialization():
    d = GCSDownloader()
    assert d.client is not None
