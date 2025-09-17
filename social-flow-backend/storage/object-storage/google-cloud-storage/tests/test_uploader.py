import pytest
from storage.object-storage.google-cloud-storage.uploader import GCSUploader


def test_upload_directory(monkeypatch, tmp_path):
    uploader = GCSUploader()

    calls = []
    monkeypatch.setattr(uploader, "upload_file", lambda p, k: calls.append((p, k)))

    f = tmp_path / "test.txt"
    f.write_text("hello")

    uploader.upload_directory(tmp_path)
    assert len(calls) == 1
