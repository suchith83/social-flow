# Exporters (file, ES, Kafka adapters)
"""
Pluggable exporters for structured logs:
- FileExporter (ndjson)
- SimpleElasticsearchExporter (bulk style; placeholder for real ES client)
- KafkaExporter (placeholder for confluent-kafka or aiokafka)

These exporters implement simple 'export(batch_of_json_lines)' method to make them
interchangeable within ingestion pipelines.
"""

from abc import ABC, abstractmethod
from typing import Iterable, List
from pathlib import Path
import os
from .config import CONFIG


class Exporter(ABC):
    @abstractmethod
    def export(self, ndjson_lines: Iterable[str]) -> None:
        raise NotImplementedError


class FileExporter(Exporter):
    def __init__(self, path: Path = None):
        cfg = CONFIG["EXPORTERS"]["file"]
        self.path = Path(path) if path else Path(cfg["path"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def export(self, ndjson_lines):
        for line in ndjson_lines:
            # ensure newline
            if not line.endswith("\n"):
                line = line + "\n"
            self._file.write(line)
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


class SimpleElasticsearchExporter(Exporter):
    """
    Very small placeholder exporter which writes bulk-like payloads to disk.
    Replace with elasticsearch-py client in production.
    """
    def __init__(self, index_prefix: str = None):
        cfg = CONFIG["EXPORTERS"]["elasticsearch"]
        self.hosts = cfg["hosts"]
        self.index_prefix = index_prefix or cfg["index_prefix"]

    def export(self, ndjson_lines):
        # convert ndjson to pseudo-bulk and write to local file for demo
        out = []
        for line in ndjson_lines:
            out.append('{"index":{}}')
            out.append(line.strip())
        # in production you'd call es.bulk(body="\n".join(out) + "\n")
        # Here we just write a file per call for easy debugging
        p = Path("es_bulk_out")
        p.mkdir(exist_ok=True)
        idx_file = p / f"bulk_{int(__import__('time').time())}.ndjson"
        idx_file.write_text("\n".join(out) + "\n", encoding="utf-8")


class KafkaExporter(Exporter):
    """
    Minimal placeholder: buffer lines to files named by topic.
    Replace with confluent-kafka or aiokafka in production.
    """
    def __init__(self, topic: str = None):
        cfg = CONFIG["EXPORTERS"]["kafka"]
        self.topic = topic or cfg["topic"]

    def export(self, ndjson_lines):
        out_dir = Path("kafka_out")
        out_dir.mkdir(exist_ok=True)
        f = out_dir / f"{self.topic}.ndjson"
        with open(f, "a", encoding="utf-8") as fh:
            for line in ndjson_lines:
                fh.write(line if line.endswith("\n") else line + "\n")
