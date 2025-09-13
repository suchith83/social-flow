# Build and push Docker images (local & remote registries)
# ================================================================
# File: docker_builder.py
# Purpose: Build & push Docker images with tagging, multi-platform support,
#          local build caching, and optional signing.
# Notes: This file shells out to Docker / Buildx. In production consider
#        using Docker SDK or a registry API for finer control.
# ================================================================

import os
from pathlib import Path
from typing import Optional, Dict
from utils import run_cmd, write_file, logger, unique_tag
import tempfile


class DockerBuilder:
    """
    Builds and optionally pushes Docker images for model deployments.

    Features:
    - Auto-detect Dockerfile or generate a minimal one for Python-based services
    - Buildx multi-platform builds (optional)
    - Tagging with model version, git commit SHA, and timestamp
    - Push to remote registry with basic retries
    """

    def __init__(self, workdir: str = ".", dockerfile: Optional[str] = None, registry: Optional[str] = None):
        self.workdir = Path(workdir)
        self.dockerfile = dockerfile or str(self.workdir / "Dockerfile")
        self.registry = registry.rstrip("/") if registry else None

    def generate_minimal_dockerfile(self, base_image: str = "python:3.10-slim", expose: int = 8080):
        """
        Creates a minimal production Dockerfile for a Python inference service.
        """
        content = f"""FROM {base_image}
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE {expose}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{expose}"]
"""
        write_file(self.dockerfile, content)
        logger.info("Generated minimal Dockerfile.")

    def build(self, image_name: str, tag: Optional[str] = None, push: bool = False, platforms: Optional[str] = None, build_args: Optional[Dict[str, str]] = None):
        """
        Build docker image using buildx if platforms are provided, otherwise normal docker build.
        """
        tag = tag or unique_tag(image_name)
        full_tag = f"{self.registry}/{image_name}:{tag}" if self.registry else f"{image_name}:{tag}"
        build_args_str = " ".join([f"--build-arg {k}='{v}'" for k, v in (build_args or {}).items()])

        if platforms:
            # Ensure builder exists
            run_cmd("docker buildx create --use --name model-builder || true")
            cmd = f"docker buildx build --platform {platforms} -t {full_tag} -f {self.dockerfile} {build_args_str} {self.workdir}"
            if push:
                cmd += " --push"
            else:
                cmd += " --load"
        else:
            cmd = f"docker build -t {full_tag} -f {self.dockerfile} {build_args_str} {self.workdir}"
            if push:
                cmd += f" && docker push {full_tag}"

        run_cmd(cmd)
        logger.info(f"Built image {full_tag}")
        return full_tag

    def push(self, full_tag: str):
        """Push a pre-built image tag"""
        return run_cmd(f"docker push {full_tag}")
