# scripts/deployment/docker_manager.py
import logging
from typing import Dict, Any

from .utils import run_command


class DockerManager:
    """
    Handles Docker image build and push operations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo = config["docker"].get("repository")
        self.tag = config["docker"].get("tag", "latest")

    def build(self):
        logging.info("🐳 Building Docker image...")
        run_command(
            ["docker", "build", "-t", f"{self.repo}:{self.tag}", "."]
        )
        logging.info("✅ Docker build complete")

    def push(self):
        logging.info("📤 Pushing Docker image...")
        run_command(["docker", "push", f"{self.repo}:{self.tag}"])
        logging.info("✅ Docker image pushed successfully")
