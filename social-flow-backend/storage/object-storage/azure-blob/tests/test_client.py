import pytest
from storage.object-storage.azure-blob.client import AzureBlobClient


def test_client_initialization():
    client = AzureBlobClient()
    assert client.client is not None
    assert client.container_client() is not None
