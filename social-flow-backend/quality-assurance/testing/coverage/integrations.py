"""
CI/CD Integrations for coverage reports.
"""

import os
import logging
import requests
from .config import CONFIG
from .exceptions import CoverageIntegrationError

logger = logging.getLogger("coverage-integrations")


class CoverageIntegrations:
    """Integrate coverage results with GitHub, GitLab, etc."""

    def __init__(self, config=CONFIG):
        self.config = config

    def upload_github_status(self, percent: float):
        """Upload coverage status to GitHub (requires token & API)."""
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPOSITORY")
        sha = os.getenv("GITHUB_SHA")

        if not all([token, repo, sha]):
            raise CoverageIntegrationError("Missing GitHub environment variables.")

        url = f"https://api.github.com/repos/{repo}/statuses/{sha}"
        state = "success" if percent >= self.config.fail_under else "failure"
        description = f"Coverage: {percent}% (min {self.config.fail_under}%)"
        data = {"state": state, "description": description, "context": "coverage"}

        resp = requests.post(url, headers={"Authorization": f"token {token}"}, json=data)
        if resp.status_code >= 300:
            raise CoverageIntegrationError(f"GitHub status update failed: {resp.text}")
        logger.info("GitHub status updated successfully.")

    def upload_gitlab_badge(self, percent: float):
        """Placeholder for GitLab badge integration."""
        logger.info(f"GitLab badge upload simulated for {percent}%.")
