"""
Dynamic Analysis (DAST) Package

Performs dynamic security testing on live applications by simulating
attacks such as XSS, SQL injection, SSRF, open redirect, etc.
"""

from .scanner import DynamicScanner
from .orchestrator import DynamicAnalysisOrchestrator
from .report_generator import DynamicReportGenerator
