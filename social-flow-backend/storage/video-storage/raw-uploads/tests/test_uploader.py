import os
import tempfile
from raw_uploads.uploader import init_upload_flow, accept_chunk_flow, finalize_upload_flow
from raw_uploads.models import InitUploadRequest

def test_upload_end_to_end(monkeypatch):
    # init
    req = InitUploadRequest(filename="small.mp4", total_bytes=6)
    resp = init_upload_flow(req, chunk_size=3)
    upload_id = resp.upload_id
    # send two chunks
    accept_chunk_flow(upload_id, 0, b"ABC")
    accept_chunk_flow(upload_id, 1, b"DEF")
    # monkeypatch storage backend to upload to local file by setting STORAGE_BACKEND to local
    # finalize
    ev = finalize_upload_flow(upload_id)
    assert ev.upload_id == upload_id
