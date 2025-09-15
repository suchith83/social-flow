"""
Deployment Pipeline
- Prepares model artifact for serving (wraps pipeline, creates versioned release)
- Optionally pushes to a model serving platform or container registry
- Exposes a `run()` to create a deployed bundle (tarball or push to S3)
"""

import os
import tarfile
from .config import settings
from .utils import logger, ensure_dir
from analytics.predictive.models.model_registry import get_model
from analytics.predictive.models.utils import save_pickle, load_pickle
import shutil
import time

class DeploymentPipeline:
    name = "deployment"

    def __init__(self, model_name: str = "user_growth_xgb", version: int | None = None, out_dir: str | None = None):
        self.model_name = model_name
        self.version = version
        self.out_dir = out_dir or os.path.join(settings.BASE_DIR, "deployments")
        ensure_dir(self.out_dir)

    def dry_run(self):
        logger.info(f"Deployment dry-run for {self.model_name} v{self.version or 'latest'}")

    def run(self):
        # fetch model registry entry (this will raise if missing)
        entry = get_model(self.model_name, version=self.version)
        artifact_path = entry["artifact_path"]
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        # create release bundle: model + metadata + requirements hint
        timestamp = int(time.time())
        bundle_name = f"{self.model_name}_v{entry['version']}_{timestamp}.tar.gz"
        bundle_path = os.path.join(self.out_dir, bundle_name)

        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(artifact_path, arcname=os.path.basename(artifact_path))
            # add metadata
            meta_path = os.path.join(self.out_dir, f"{self.model_name}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                import json
                json.dump(entry, f, indent=2)
            tar.add(meta_path, arcname=os.path.basename(meta_path))
            # optionally include a requirements.txt hint
            req_hint = os.path.join(self.out_dir, "requirements_hint.txt")
            with open(req_hint, "w", encoding="utf-8") as f:
                f.write("scikit-learn\nxgboost\npandas\nnumpy\njoblib\n")
            tar.add(req_hint, arcname=os.path.basename(req_hint))

        logger.info(f"Created deployment bundle: {bundle_path}")
        # optional: push to S3 or artifact registry (hook here)
        return {"bundle_path": bundle_path, "registry_entry": entry}
