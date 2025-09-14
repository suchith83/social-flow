"""
Dynamic Analysis Orchestrator
Coordinates scanning and reporting.
"""

from .scanner import DynamicScanner
from .report_generator import DynamicReportGenerator
from .utils import logger
from .config import SCAN_CONFIG


class DynamicAnalysisOrchestrator:
    def __init__(self):
        self.scanner = DynamicScanner(SCAN_CONFIG["base_url"])
        self.reporter = DynamicReportGenerator()

    def run_scan(self, endpoints: list):
        logger.info(f"Starting DAST scan on {SCAN_CONFIG['base_url']}")
        results = self.scanner.scan(endpoints)
        report_path = self.reporter.generate(results)
        logger.info(f"DAST pipeline completed. Report: {report_path}")
        return report_path
