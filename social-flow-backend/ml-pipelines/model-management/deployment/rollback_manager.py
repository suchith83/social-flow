# Safe rollback strategies and record keeping
# ================================================================
# File: rollback_manager.py
# Purpose: Manage safe rollbacks:
#  - Record deployment metadata in a local JSON log store
#  - Determine previous stable replica set or image tag
#  - Trigger rollback via kubectl or Helm
# ================================================================

import json
from pathlib import Path
from utils import write_file, run_cmd, logger, timestamp

LOG_PATH = Path("deployments/deploy_history.json")


class RollbackManager:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self._write([])

    def _read(self):
        return json.loads(self.log_path.read_text())

    def _write(self, obj):
        self.log_path.write_text(json.dumps(obj, indent=2))

    def record(self, name: str, image: str, namespace: str = "default", metadata: dict = None):
        history = self._read()
        entry = {
            "time": timestamp(),
            "name": name,
            "image": image,
            "namespace": namespace,
            "metadata": metadata or {}
        }
        history.append(entry)
        self._write(history)
        logger.info(f"Recorded deployment: {entry}")

    def history(self, limit: int = 10):
        return self._read()[-limit:]

    def rollback_to(self, name: str, image: str = None, namespace: str = "default"):
        """
        If image provided rollback to that image. Otherwise rollback to previous entry for `name`.
        """
        history = self._read()
        candidates = [h for h in history if h["name"] == name]
        if not candidates:
            raise RuntimeError(f"No deployment history for {name}")
        target = None
        if image:
            for c in reversed(candidates):
                if c["image"] == image:
                    target = c
                    break
            if not target:
                raise RuntimeError(f"Image {image} not found in history for {name}")
        else:
            if len(candidates) < 2:
                raise RuntimeError("No previous deployment to rollback to")
            target = candidates[-2]

        image_to_use = target["image"]
        logger.info(f"Rolling back {name} in namespace {namespace} to image {image_to_use}")
        # Use kubectl set image pattern - assumes deployment name == service name
        cmd = f"kubectl -n {namespace} set image deployment/{name} {name}-container={image_to_use}"
        run_cmd(cmd)
        logger.info("Rollback applied.")
        return target
