import pytest
from storage.object-storage.aws-s3.client import S3Client


def test_client_initialization():
    client = S3Client()
    assert client.client is not None
    assert client.resource() is not None
