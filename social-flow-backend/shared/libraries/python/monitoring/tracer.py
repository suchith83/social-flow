# tracer.py
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Any


class Tracer:
    """
    Simple distributed tracing utility.
    Compatible with OpenTelemetry exporters.
    """

    def __init__(self):
        self.spans = []

    @contextmanager
    def start_span(self, name: str, parent_id: str = None):
        """Context manager to create a span."""
        span_id = str(uuid.uuid4())
        start_time = time.time()
        yield {"span_id": span_id, "parent_id": parent_id, "name": name}
        duration = time.time() - start_time
        self.spans.append({
            "span_id": span_id,
            "parent_id": parent_id,
            "name": name,
            "duration": duration,
            "timestamp": start_time
        })

    def export(self) -> Dict[str, Any]:
        """Export spans for reporting."""
        return {"traces": self.spans}
