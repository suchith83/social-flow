# common/libraries/python/ml/experiment.py
"""
Experiment tracking utilities.
Logs metrics, configs, artifacts.
"""

import os
import json
from datetime import datetime
from .config import MLConfig

class ExperimentTracker:
    def __init__(self, name: str):
        self.name = name
        self.exp_dir = os.path.join(MLConfig.EXPERIMENTS_DIR, name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.log_file = os.path.join(self.exp_dir, "log.jsonl")

    def log_metrics(self, metrics: dict):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_params(self, params: dict):
        with open(os.path.join(self.exp_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

    def save_artifact(self, filename: str, content: bytes):
        path = os.path.join(self.exp_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        return path
