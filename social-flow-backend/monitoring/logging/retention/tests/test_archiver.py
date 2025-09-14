# Tests for archiver module
# monitoring/logging/retention/tests/test_archiver.py
from pathlib import Path
from monitoring.logging.retention.archiver import Archiver

def test_archiver(tmp_path):
    file = tmp_path / "test.log"
    file.write_text("sample log")
    arch = Archiver()
    archived = arch.archive(file)
    assert archived.exists()
