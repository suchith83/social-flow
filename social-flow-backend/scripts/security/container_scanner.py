# scripts/security/container_scanner.py
import logging
import json
from typing import Dict, Any, List, Optional

from .utils import which, run_command, write_json

logger = logging.getLogger("security.container")


class ContainerScanner:
    """
    Scan container images using Trivy if available. Falls back to 'docker scan' (Snyk) if present.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("security", {}).get("container", {})
        self.images = self.cfg.get("images", [])
        self.scanner = self.cfg.get("scanner", "trivy")
        self.output_dir = config.get("security", {}).get("reporter", {}).get("output_dir", "./security-reports")
        self.enabled = bool(self.cfg.get("enabled", True))

    def trivy_scan(self, image: str) -> Dict[str, Any]:
        if not which("trivy"):
            logger.warning("trivy not found")
            return {"tool": None, "note": "trivy-not-found"}
        cmd = ["trivy", "image", "--quiet", "--format", "json", image]
        rc, out, err = run_command(cmd)
        try:
            data = json.loads(out or "{}")
        except Exception:
            data = {"raw": out, "err": err}
        report = {"tool": "trivy", "image": image, "data": data}
        write_json(f"{self.output_dir}/container-trivy-{image.replace('/', '_').replace(':', '_')}.json", report)
        return report

    def docker_scan(self, image: str) -> Dict[str, Any]:
        # docker scan uses Snyk; output parsing is not standardized here
        if not which("docker"):
            return {"tool": None, "note": "docker-not-found"}
        cmd = ["docker", "scan", "--json", image]
        rc, out, err = run_command(cmd)
        try:
            data = json.loads(out or "{}")
        except Exception:
            data = {"raw": out, "err": err}
        report = {"tool": "docker-scan", "image": image, "data": data}
        write_json(f"{self.output_dir}/container-docker-scan-{image.replace('/', '_').replace(':', '_')}.json", report)
        return report

    def run(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            logger.info("Container scanning disabled")
            return []
        results = []
        for img in self.images:
            try:
                if self.scanner == "trivy":
                    results.append(self.trivy_scan(img))
                else:
                    results.append(self.docker_scan(img))
            except Exception:
                logger.exception("Container scan failed for %s", img)
        return results
