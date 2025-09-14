# Tests for serializer & schema
import pytest
from monitoring.logging.structured.schema import StructuredLog
from monitoring.logging.structured.serializer import to_ndjson_line, from_json_str
from datetime import datetime

def test_schema_basic():
    s = StructuredLog(timestamp=datetime.utcnow(), service="svc", level="info", message="ok")
    assert s.service == "svc"
    assert s.level == "INFO"

def test_serialize_roundtrip():
    s = StructuredLog(timestamp=datetime.utcnow(), service="svc", level="error", message="oops")
    nd = to_ndjson_line(s)
    obj = from_json_str(nd)
    assert obj["service"] == "svc"
    assert obj["level"] == "ERROR"
