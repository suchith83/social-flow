import os
import tempfile
from raw_uploads.chunk_manager import make_upload_id, init_upload, write_chunk, assemble, cleanup_upload, is_complete

def test_chunk_flow():
    uid = make_upload_id()
    filename = "test.mp4"
    # small test file bytes
    chunks = [b"AAA", b"BBB", b"CCC"]
    total = sum(len(c) for c in chunks)
    init_upload(uid, filename, total, chunk_size=3)
    for i, c in enumerate(chunks):
        n = write_chunk(uid, i, c)
        assert n == len(c)
    assert is_complete(uid)
    staged = assemble(uid)
    assert os.path.exists(staged)
    with open(staged, "rb") as f:
        data = f.read()
    assert data == b"AAABBBCCC"
    cleanup_upload(uid)
    assert not os.path.exists(staged)
