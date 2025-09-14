# Configurations (paths, exporters, serialization settings)
"""
Configuration for structured logging module.
Edit values to adapt to your environment (paths, exporter defaults, etc.)
"""
from pathlib import Path

CONFIG = {
    "DEFAULT_SERVICE_NAME": "unknown-service",
    "SERIALIZATION": {
        "ndjson_batch_size": 500,
        "ensure_ascii": False,
        "indent": None,
    },
    "EXPORTERS": {
        "file": {
            "path": Path("logs/structured.ndjson"),
            "rotate_size_bytes": 10 * 1024 * 1024,  # 10MB
        },
        # placeholders for real backends; replace with real clients in production
        "elasticsearch": {
            "hosts": ["http://localhost:9200"],
            "index_prefix": "app-logs-",
        },
        "kafka": {
            "bootstrap_servers": ["localhost:9092"],
            "topic": "structured-logs",
        }
    },
    "VALIDATION": {
        "max_message_length": 32_768,  # protect against huge messages
    }
}
