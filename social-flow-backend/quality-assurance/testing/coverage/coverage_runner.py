"""
Coverage runner - orchestrates execution of coverage measurement.
"""

import subprocess
import logging
from .config import CONFIG
from .utils import ensure_clean_dir
from .exceptions import CoverageError

logger = logging.getLogger("coverage-runner")


class CoverageRunner:
    """Run coverage analysis with pytest + coverage.py."""

    def __init__(self, config=CONFIG):
        self.config = config

    def run(self):
        """Execute tests with coverage."""
        ensure_clean_dir(self.config.report_dir)
        logger.info("Running coverage...")

        try:
            cmd = [
                "coverage", "run",
                "--source", ",".join(self.config.source_dirs),
                "-m", "pytest"
            ]
            subprocess.run(cmd, check=True)

            if self.config.xml_report:
                subprocess.run(["coverage", "xml", "-o", f"{self.config.report_dir}/coverage.xml"], check=True)

            if self.config.json_report:
                subprocess.run(["coverage", "json", "-o", f"{self.config.report_dir}/coverage.json"], check=True)

            if self.config.html_report:
                subprocess.run(["coverage", "html", "-d", f"{self.config.report_dir}/html"], check=True)

            logger.info("Coverage execution finished successfully.")
        except subprocess.CalledProcessError as e:
            raise CoverageError(f"Coverage execution failed: {e}")
