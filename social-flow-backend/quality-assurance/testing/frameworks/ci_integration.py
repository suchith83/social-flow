"""
CI/CD integration helpers for test frameworks:
- Upload reports to CI artifacts (placeholder)
- Set commit status for test pass/fail (GitHub/GitLab)
- Fail build when required thresholds not met
"""

import os
import logging
import json
from typing import Any, Dict, List

from .config import FRAMEWORKS_CONFIG

logger = logging.getLogger("qa-testing-frameworks.ci")


class CIIntegration:
    """
    Small helpers to integrate generated reports with CI systems.
    The actual transport (uploading artifacts or posting statuses) depends on
    environment variables and credentials available in the CI environment.
    """

    def __init__(self, config=FRAMEWORKS_CONFIG):
        self.config = config

    def publish_artifact(self, path: str):
        """
        Attempt to publish artifact to CI. Implementations:
          - GitHub Actions: use 'GITHUB_ACTIONS' env and write to $GITHUB_WORKSPACE or allow artifact upload.
          - GitLab CI: artifacts are picked up by pipeline if stored in reports/ and configured in .gitlab-ci.yml.
        This function will act conservatively: if known CI env vars exist, print hints.
        """
        if os.getenv("GITHUB_ACTIONS") == "true":
            logger.info("Running on GitHub Actions. Place artifact at %s and configure actions/upload-artifact to pick it up.", path)
        elif os.getenv("GITLAB_CI") == "true":
            logger.info("Running on GitLab CI. Ensure 'artifacts:paths' includes %s in .gitlab-ci.yml.", path)
        else:
            logger.info("CI artifact publish: no known CI detected. Artifact left at %s", path)

    def set_status(self, success: bool, description: str = ""):
        """
        Set commit/check status. This is an advisory helper — concrete implementation
        requires tokens/environment variables. We log the action and return a dict for callers to implement.
        """
        status = "success" if success else "failure"
        payload = {"state": status, "description": description}
        if os.getenv("GITHUB_ACTIONS") == "true":
            # Suggest how to set status using GitHub API (token/GITHUB_REPOSITORY required)
            logger.info("To set GitHub commit status, POST to /repos/{owner}/{repo}/statuses/{sha} with a token. Payload: %s", payload)
        elif os.getenv("GITLAB_CI") == "true":
            logger.info("To set GitLab commit status, use the Statuses API with CI_JOB_TOKEN. Payload: %s", payload)
        else:
            logger.info("No CI detected; status would be: %s", payload)
        return payload

    def enforce_policy(self, aggregated_report: Dict[str, Any], fail_on_failure: bool = True) -> bool:
        """
        Enforce policy rules for tests: for now fail when aggregated summary reports any failures.
        Return True when policy passes, False otherwise.
        """
        summary = aggregated_report.get("summary", {})
        failed = summary.get("failed", 0)
        if failed > 0:
            logger.warning("Enforcement: detected %s failed tests.", failed)
            if fail_on_failure:
                raise RuntimeError(f"Test policy enforcement failed: {failed} failed tests")
            return False
        logger.info("Enforcement: tests passed")
        return True
