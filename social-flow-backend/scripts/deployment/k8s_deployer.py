# scripts/deployment/k8s_deployer.py
import logging
import subprocess
from typing import Dict, Any

from .utils import run_command


class K8sDeployer:
    """
    Deploys services to Kubernetes using kubectl or Helm.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.namespace = config["k8s"].get("namespace", "default")

    def deploy(self):
        logging.info("🚀 Deploying to Kubernetes...")
        manifests = self.config["k8s"].get("manifests", [])
        for manifest in manifests:
            run_command(["kubectl", "apply", "-f", manifest, "-n", self.namespace])
        logging.info("✅ Kubernetes deployment complete")

    def rollback(self, deployment: str):
        logging.warning(f"⚠️ Rolling back deployment {deployment}...")
        run_command(
            ["kubectl", "rollout", "undo", f"deployment/{deployment}", "-n", self.namespace]
        )
        logging.info(f"✅ Rollback complete for {deployment}")
