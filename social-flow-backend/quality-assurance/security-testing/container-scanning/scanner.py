"""
Container Scanner
Analyzes container images for vulnerabilities and misconfigurations.
"""

import docker
from .vulnerability_database import VulnerabilityDatabase
from .utils import logger


class ContainerScanner:
    def __init__(self):
        self.client = docker.from_env()
        self.vuln_db = VulnerabilityDatabase()

    def pull_image(self, image: str):
        """Pull container image."""
        logger.info(f"Pulling image: {image}")
        self.client.images.pull(image)
        return image

    def scan_image(self, image: str) -> dict:
        """
        Inspect image layers and scan installed packages against vulnerability DB.
        """
        self.pull_image(image)
        img = self.client.images.get(image)
        history = img.history()

        findings = {
            "image": image,
            "vulnerabilities": [],
            "metadata": {"id": img.id, "tags": img.tags}
        }

        # Fake package extraction (would use Trivy/Grype/Clair integration in prod)
        sample_packages = [("openssl", "1.1.1"), ("nginx", "1.21.0")]

        for pkg, version in sample_packages:
            vulns = self.vuln_db.find_vulnerabilities(pkg, version)
            if vulns:
                findings["vulnerabilities"].extend(vulns)

        logger.info(f"Scan completed for {image} with {len(findings['vulnerabilities'])} vulns found.")
        return findings
