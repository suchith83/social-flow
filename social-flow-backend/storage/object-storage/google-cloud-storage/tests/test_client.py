import pytest
from storage.object-storage.google-cloud-storage.client import GCSClient


def test_client_initialization():
    client = GCSClient()
    assert client.client is not None
    assert client.bucket is not None
