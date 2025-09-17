import pytest
from storage.object-storage.aws-s3.uploader import S3Uploader


def test_upload_directory(monkeypatch, tmp_path):
    uploader = S3Uploader()

    calls = []
    monkeypatch.setattr(uploader, "upload_file", lambda p, k: calls.append((p, k)))

    # create dummy files
    f = tmp_path / "test.txt"
    f.write_text("hello")

    uploader.upload_directory(tmp_path)
    assert len(calls) == 1
