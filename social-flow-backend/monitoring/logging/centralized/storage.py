# Storage engine (in-memory + pluggable backends like ES/DB)
# monitoring/logging/centralized/storage.py
"""
Storage engine for centralized logs.
Implements in-memory and file-based backends, extensible to DB/ES.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from .config import CONFIG
from .utils import stable_id


class InMemoryStorage:
    def __init__(self):
        self.logs = {}

    def store(self, logs: list):
        for log in logs:
            log_id = stable_id(log)
            self.logs[log_id] = log

    def retrieve(self, log_id: str):
        return self.logs.get(log_id)

    def query_all(self):
        return list(self.logs.values())

    def purge_old(self):
        retention = timedelta(days=CONFIG["STORAGE"]["retention_days"])
        cutoff = datetime.utcnow() - retention
        self.logs = {
            k: v for k, v in self.logs.items()
            if v["timestamp"] and v["timestamp"] > cutoff
        }


class FileStorage:
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def store(self, logs: list):
        for log in logs:
            log_id = stable_id(log)
            file = self.path / f"{log_id}.json"
            with open(file, "w") as f:
                json.dump(log, f, default=str)

    def query_all(self):
        all_logs = []
        for file in self.path.glob("*.json"):
            with open(file) as f:
                all_logs.append(json.load(f))
        return all_logs
