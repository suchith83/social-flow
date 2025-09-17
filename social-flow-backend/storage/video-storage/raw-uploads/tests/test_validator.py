import os
import tempfile
from raw_uploads.validator import validate_staged_file, UploadValidationError

def test_validate_small_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        tf.write(b"\x00\x00\x00")
        path = tf.name
    try:
        # this will raise because mime sniff may fail or size mismatch if total provided
        r = validate_staged_file(path, expected_bytes=3, mime_override="video/mp4")
        assert r["size"] == 3
    finally:
        os.remove(path)
