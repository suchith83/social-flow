"""
Configuration for test frameworks: default paths, timeouts and toggles.
Environment variables override defaults.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FrameworksConfig:
    tests_dir: str = os.getenv("TESTS_DIR", "tests")
    junit_xml_path: str = os.getenv("JUNIT_XML_PATH", "reports/junit.xml")
    json_report_path: str = os.getenv("JSON_REPORT_PATH", "reports/tests.json")
    verbose: bool = os.getenv("TESTS_VERBOSE", "1") == "1"
    fail_fast: bool = os.getenv("TESTS_FAIL_FAST", "0") == "1"
    timeout_seconds: int = int(os.getenv("TEST_TIMEOUT", "300"))
    selected_frameworks: List[str] = os.getenv("TEST_FRAMEWORKS", "pytest").split(",")  # e.g. "pytest,unittest"

    @staticmethod
    def from_env():
        return FrameworksConfig()


FRAMEWORKS_CONFIG = FrameworksConfig.from_env()
