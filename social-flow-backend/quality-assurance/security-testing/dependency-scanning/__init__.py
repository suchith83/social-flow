"""
Dependency Scanning Package

Provides scanning of project dependencies for known vulnerabilities
across multiple ecosystems (Python, Node.js, Java).
"""

from .scanner import DependencyScanner
from .orchestrator import DependencyScanOrchestrator
from .report_generator import DependencyReportGenerator
