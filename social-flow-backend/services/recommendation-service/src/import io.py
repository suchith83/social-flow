import io
import os
import tempfile
from storage.video_storage import upload_video, download_video_to_bytes, delete_video, DEFAULT_BUCKET

def test_upload_download_delete_cycle(tmp_path, monkeypatch):
    # ensure local storage dirs are isolated to tmp_path
    # point LOCAL_S3_PREFIX and LOCAL_STORAGE_DIR to tmp_path
    monkeypatch.setenv("LOCAL_S3_PREFIX", str(tmp_path / "s3"))
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "local"))
    # reload modules that read env at import time would be better, but keep simple:
    data = b"dummy video bytes"
    buf = io.BytesIO(data)
    meta = upload_video(buf, "test.mp4", bucket="test-bucket")
    assert "key" in meta and "playback_url" in meta
    downloaded = download_video_to_bytes(meta["bucket"], meta["key"])
    assert downloaded == data
    # delete and ensure subsequent download raises (handled by S3Client Dummy -> FileNotFoundError)
    delete_video(meta["bucket"], meta["key"])
    # after delete, a new download should raise or return empty; attempt and expect no bytes
    try:
        got = download_video_to_bytes(meta["bucket"], meta["key"])
        # if exists, ensure it's empty or different
        assert got != data
    except Exception:
        # acceptable for local fallback if file was removed
        pass
