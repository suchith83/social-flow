"""
Scan Orchestrator
Coordinates scanning and reporting workflows.
"""

from .scanner import ContainerScanner
from .report_generator import ReportGenerator
from .utils import logger


class ScanOrchestrator:
    def __init__(self):
        self.scanner = ContainerScanner()
        self.reporter = ReportGenerator()

    def run_scan(self, image: str):
        """Run full scan pipeline for an image."""
        logger.info(f"Starting scan pipeline for {image}")
        results = self.scanner.scan_image(image)
        report_path = self.reporter.generate(results)
        logger.info(f"Pipeline completed. Report: {report_path}")
        return report_path
