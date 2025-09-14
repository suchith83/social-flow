"""
Dependency Scan Orchestrator
"""

from .parser import DependencyParser
from .scanner import DependencyScanner
from .report_generator import DependencyReportGenerator
from .utils import logger


class DependencyScanOrchestrator:
    def __init__(self):
        self.scanner = DependencyScanner()
        self.reporter = DependencyReportGenerator()

    def run_scan(self, ecosystem: str, manifest_path: str):
        logger.info(f"Starting dependency scan for {ecosystem} project")
        if ecosystem == "python":
            deps = DependencyParser.parse_python(manifest_path)
        elif ecosystem == "node":
            deps = DependencyParser.parse_node(manifest_path)
        elif ecosystem == "java":
            deps = DependencyParser.parse_java(manifest_path)
        else:
            raise ValueError(f"Unsupported ecosystem: {ecosystem}")

        results = self.scanner.scan(ecosystem, deps)
        report = self.reporter.generate(results)
        logger.info(f"Dependency scan completed. Report at {report}")
        return report
