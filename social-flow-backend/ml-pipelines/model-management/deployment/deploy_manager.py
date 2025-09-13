# High-level deployment orchestration
# ================================================================
# File: deploy_manager.py
# Purpose: High-level deployment orchestrator that ties the above pieces together.
#
# Features:
#  - Build/push image
#  - Generate manifests/helm charts
#  - Apply manifests (kubectl) or helm upgrade --install
#  - Record deployment metadata for rollbacks
#  - Optionally run health-check verification
# ================================================================

import os
from docker_builder import DockerBuilder
from k8s_generator import K8sGenerator
from helm_helper import generate_chart
from ci_cd import generate_workflow
from rollback_manager import RollbackManager
from monitor_integration import generate_service_monitor, generate_alert_rule
from utils import logger, run_cmd, unique_tag, write_file
from typing import Dict, Any


class DeployManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.registry = cfg.get("registry")
        self.builder = DockerBuilder(workdir=cfg.get("workdir", "."), registry=self.registry)
        self.k8s_gen = K8sGenerator(out_dir=cfg.get("manifests_dir", "manifests"))
        self.rollback = RollbackManager()
        self.namespace = cfg.get("namespace", "default")

    def build_and_push(self, image_name: str, tag: str = None, push: bool = True, build_args: Dict[str, str] = None):
        # Ensure a Dockerfile exists (or generate a minimal one)
        if not os.path.exists(self.builder.dockerfile):
            self.builder.generate_minimal_dockerfile()
        tag = tag or unique_tag(image_name)
        full_tag = self.builder.build(image_name, tag=tag, push=push, build_args=build_args)
        return full_tag

    def deploy_k8s(self, spec: Dict[str, Any], method: str = "kubectl"):
        """
        method: 'kubectl' or 'helm'
        spec: See K8sGenerator.generate expected keys
        """
        files = self.k8s_gen.generate(spec)
        if method == "kubectl":
            for path in files.keys():
                run_cmd(f"kubectl -n {self.namespace} apply -f {path}")
        elif method == "helm":
            chart_dir = generate_chart(spec["name"], out_dir=self.cfg.get("charts_dir", "charts"))
            run_cmd(f"helm upgrade --install {spec['name']} {chart_dir} -n {self.namespace} --set image.repository={spec['image'].split(':')[0]} --set image.tag={spec['image'].split(':')[-1]}")
        else:
            raise ValueError("Unknown deploy method")
        # Optionally create servicemonitor and alert rules
        if self.cfg.get("monitoring", {}).get("enabled", True):
            generate_service_monitor(spec["name"])
            generate_alert_rule(spec["name"])

        # Record for rollback
        self.rollback.record(spec["name"], spec["image"], namespace=self.namespace, metadata={"spec": spec})

    def health_check(self, host: str, path: str = "/health", timeout: int = 30) -> bool:
        """Simple curl based health check. Replace with robust HTTP client in production."""
        cmd = f"timeout {timeout} curl -fsSL -o /dev/null -w '%{{http_code}}' {host}{path}"
        try:
            status = run_cmd(cmd, check=False)
            ok = status == "200"
            logger.info(f"Health check {host}{path} -> {status} (ok={ok})")
            return ok
        except Exception:
            return False

    def deploy(self, image_name: str, build: bool = True, spec_overrides: Dict[str, Any] = None, method: str = "kubectl"):
        # Build & push
        tag = unique_tag(image_name)
        logger.info(f"Starting deploy for {image_name}:{tag}")
        if build:
            image = self.build_and_push(image_name, tag=tag, push=True)
        else:
            # assume image exists in registry with tag provided as image_name (full image)
            image = image_name

        spec = {
            "name": self.cfg.get("service_name", "model-service"),
            "model_name": self.cfg.get("model_name", "model"),
            "version": tag,
            "image": image,
            "port": self.cfg.get("port", 8080),
            "replicas": self.cfg.get("replicas", 2),
            "env": self.cfg.get("env", {}),
            "resources": self.cfg.get("resources", {}),
            "hpa": self.cfg.get("hpa", None),
            "ingress": self.cfg.get("ingress", None),
            "service_type": self.cfg.get("service_type", "ClusterIP"),
        }

        if spec_overrides:
            spec.update(spec_overrides)

        # Deploy
        self.deploy_k8s(spec, method=method)

        # Optionally health check
        ingress_host = (spec.get("ingress") or {}).get("host")
        if ingress_host:
            ok = self.health_check(f"https://{ingress_host}", path=spec.get("readiness_path", "/health"))
            if not ok:
                # Rollback automatically if health check fails
                logger.error("Health check failed; triggering rollback")
                self.rollback.rollback_to(spec["name"])
                raise RuntimeError("Deployment failed health check and rollback triggered")

        logger.info("Deployment finished successfully.")
        return spec
