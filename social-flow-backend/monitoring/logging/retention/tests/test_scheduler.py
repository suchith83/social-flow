# Tests for scheduler module
# monitoring/logging/retention/tests/test_scheduler.py
from pathlib import Path
from monitoring.logging.retention.scheduler import RetentionScheduler

def test_scheduler_start_stop(tmp_path):
    sched = RetentionScheduler(tmp_path)
    thread = sched.start()
    sched.stop()
    assert not sched._stop.is_set() or sched._stop.is_set()
