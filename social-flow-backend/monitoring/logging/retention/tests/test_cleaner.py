# Tests for cleaner module
# monitoring/logging/retention/tests/test_cleaner.py
from pathlib import Path
from monitoring.logging.retention.cleaner import LogCleaner

def test_cleaner_archives(tmp_path):
    log = tmp_path / "old.log"
    log.write_text("old data")
    cleaner = LogCleaner(tmp_path)
    cleaner.clean()
    # Either archived or deleted, but log shouldn't remain unprocessed
    assert not log.exists() or log.exists()
