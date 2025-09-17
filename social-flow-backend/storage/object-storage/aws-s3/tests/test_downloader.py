import pytest
from storage.object-storage.aws-s3.downloader import S3Downloader


def test_downloader_initialization():
    d = S3Downloader()
    assert d.client is not None
