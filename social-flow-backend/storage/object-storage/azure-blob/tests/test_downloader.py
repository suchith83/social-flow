import pytest
from storage.object-storage.azure-blob.downloader import AzureBlobDownloader


def test_downloader_initialization():
    d = AzureBlobDownloader()
    assert d.client is not None
