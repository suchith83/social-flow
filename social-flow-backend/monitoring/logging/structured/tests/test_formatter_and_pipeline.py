# Tests for formatter & ingest pipeline
import logging
import tempfile
from monitoring.logging.structured.formatter import StructuredJSONFormatter
from monitoring.logging.structured.serializer import from_json_str
from monitoring.logging.structured.ingest_pipeline import IngestPipeline
from monitoring.logging.structured.exporter import FileExporter
from pathlib import Path

def test_formatter_writes_ndjson(tmp_path):
    f = tmp_path / "slog.ndjson"
    fh = logging.FileHandler(str(f))
    fmt = StructuredJSONFormatter(service="testsvc")
    fh.setFormatter(fmt)
    logger = logging.getLogger("structured.test")
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("hello %s", "world")
    fh.close()

    text = f.read_text()
    assert "hello world" in text
    obj = from_json_str(text.strip().splitlines()[-1])
    assert obj["service"] == "testsvc"

def test_ingest_pipeline_roundtrip(tmp_path):
    out_file = tmp_path / "pipe.ndjson"
    exporter = FileExporter(path=out_file)
    pipe = IngestPipeline(exporter=exporter, batch_size=2)
    inputs = [
        {"service": "s1", "message": "m1"},
        {"service": "s2", "message": "m2"},
        {"service": "s3", "message": "m3"}
    ]
    pipe.process_iterable(inputs)
    txt = out_file.read_text()
    assert "m1" in txt and "m2" in txt and "m3" in txt
