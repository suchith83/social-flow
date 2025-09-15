"""
Pipeline Orchestrator
- Provides a simple programmatic orchestrator that can sequentially run
  pipeline stages and handle failures, checkpoints, and idempotency.
- Each pipeline stage is expected to be a class with `run()` and `dry_run()` methods.
"""

from typing import List, Any
from .utils import logger, write_json, ensure_dir
import os
from .config import settings
import time


class Orchestrator:
    def __init__(self, name: str = "default-run"):
        self.name = name
        self.checkpoint_dir = os.path.join(settings.BASE_DIR, "runs", self.name)
        ensure_dir(self.checkpoint_dir)

    def _checkpoint_path(self, stage_name: str) -> str:
        return os.path.join(self.checkpoint_dir, f"{stage_name}.done")

    def stage_completed(self, stage_name: str) -> bool:
        return os.path.exists(self._checkpoint_path(stage_name))

    def mark_completed(self, stage_name: str, metadata: dict | None = None):
        path = self._checkpoint_path(stage_name)
        write_json(path, {"completed_at": time.time(), "meta": metadata or {}})

    def run(self, stages: List[Any], resume: bool = True):
        """
        Run pipeline stages sequentially. Each stage must be an object with:
            - name: str
            - run(): executes stage
            - dry_run(): optional, to check readiness
        resume: if True, will skip stages with checkpoint files
        """
        logger.info(f"Starting orchestration run: {self.name}")
        for stage in stages:
            name = getattr(stage, "name", stage.__class__.__name__)
            logger.info(f"Processing stage: {name}")
            if resume and self.stage_completed(name):
                logger.info(f"Skipping {name} (checkpoint exists)")
                continue
            try:
                # optional dry run
                if hasattr(stage, "dry_run"):
                    logger.info(f"Dry-run stage: {name}")
                    stage.dry_run()
                # run
                start = time.time()
                result = stage.run()
                duration = time.time() - start
                logger.info(f"Stage {name} completed in {duration:.2f}s")
                self.mark_completed(name, {"duration_s": duration})
            except Exception as exc:
                logger.exception(f"Stage {name} failed: {exc}")
                raise
        logger.info("Orchestration completed")
