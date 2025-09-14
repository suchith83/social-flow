# Tests for log_parser
# monitoring/logging/analysis/tests/test_parser.py
import pytest
from monitoring.logging.analysis.log_parser import LogParser

def test_json_parser():
    parser = LogParser()
    log = parser.parse('{"timestamp": "2023-01-01 12:00:00", "msg": "ok"}')
    assert "msg" in log

def test_syslog_parser():
    parser = LogParser()
    raw = "Jan 12 12:00:00 localhost system rebooted"
    log = parser.parse(raw)
    assert "message" in log
