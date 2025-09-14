# Ingestion pipeline (validate ? transform ? serialize ? export)
"""
A simple ingestion pipeline that validates, transforms, serializes and exports logs.
It supports batching for efficiency.
"""

from typing import Iterable, Dict, Any, List
from .validator import validate
from .transformer import enrich
from .serializer import to_ndjson_line
from .utils import ndjson_batches
from .exporter import FileExporter, Exporter
from .config import CONFIG


class IngestPipeline:
    def __init__(self, exporter: Exporter = None, batch_size: int = None):
        self.exporter = exporter or FileExporter()
        self.batch_size = batch_size or CONFIG["SERIALIZATION"]["ndjson_batch_size"]

    def process_iterable(self, inputs: Iterable[Dict[str, Any]]):
        """
        Process an iterable of unstructured dict-like logs.

        Steps:
        1. Enrich
        2. Validate (attempt to coerce to StructuredLog)
        3. Serialize to NDJSON
        4. Export in batches
        """
        ndjson_lines = []
        for raw in inputs:
            try:
                enriched = enrich(raw)
                model, err = validate(enriched)
                if err:
                    # If invalid, attach validation error to attrs and continue with best-effort payload
                    enriched.setdefault("attrs", {})["validation_error"] = str(err)
                    nd = to_ndjson_line(enriched)
                else:
                    nd = to_ndjson_line(model)
                ndjson_lines.append(nd)
            except Exception as e:
                # Log serialization failure as a minimal fallback line
                fallback = {"timestamp": None, "service": CONFIG["DEFAULT_SERVICE_NAME"], "level": "ERROR", "message": f"ingest_failure: {e}"}
                ndjson_lines.append(to_ndjson_line(fallback))

            # flush if batch size reached
            if len(ndjson_lines) >= self.batch_size:
                self.exporter.export(ndjson_lines)
                ndjson_lines = []

        if ndjson_lines:
            self.exporter.export(ndjson_lines)
