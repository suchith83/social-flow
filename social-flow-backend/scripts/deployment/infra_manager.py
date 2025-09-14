# scripts/deployment/infra_manager.py
import subprocess
import logging
from typing import Dict, Any

from .utils import run_command


class InfraManager:
    """
    Manages cloud infrastructure using IaC tools like Terraform or AWS CDK.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.terraform_dir = config["infra"].get("terraform_dir", "./infra")

    def provision(self):
        logging.info("🚀 Provisioning infrastructure...")
        run_command(["terraform", "init"], cwd=self.terraform_dir)
        run_command(["terraform", "apply", "-auto-approve"], cwd=self.terraform_dir)
        logging.info("✅ Infrastructure provisioned successfully")

    def destroy(self):
        logging.warning("⚠️ Destroying infrastructure...")
        run_command(["terraform", "destroy", "-auto-approve"], cwd=self.terraform_dir)
        logging.info("✅ Infrastructure destroyed successfully")
