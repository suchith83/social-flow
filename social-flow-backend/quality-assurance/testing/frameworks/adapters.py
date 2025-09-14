"""
Framework adapters — wrap pytest and unittest to expose a common interface.

Design goals:
- Allow programmatic invocation of frameworks with consistent inputs/outputs.
- Collect structured results (dict) that reporters can consume.
- Keep adapters lightweight and robust to missing dependencies.
"""

from typing import Dict, Any, List, Optional
import subprocess
import sys
import os
import tempfile
import json
import logging

from .config import FRAMEWORKS_CONFIG
from .exceptions import AdapterNotFoundError, TestExecutionError
from .utils import write_json_atomic

logger = logging.getLogger("qa-testing-frameworks.adapters")


class BaseAdapter:
    """Abstract adapter interface."""

    def __init__(self, config=FRAMEWORKS_CONFIG):
        self.config = config

    def run(self, tests: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run tests and return structured results."""
        raise NotImplementedError()


class PytestAdapter(BaseAdapter):
    """
    Adapter for pytest. Runs pytest programmatically via subprocess to keep
    plugin isolation and avoid interfering with host interpreter state.
    Produces JUnit XML and JSON summary (via pytest's --json-report if available).
    """

    def _pytest_available(self) -> bool:
        try:
            import pytest  # noqa: F401
            return True
        except Exception:
            return False

    def run(self, tests: Optional[List[str]] = None) -> Dict[str, Any]:
        # Build command
        if not self._pytest_available():
            # Fall back to invoking python -m pytest which is likely present
            logger.warning("pytest not importable; attempting python -m pytest via subprocess")
        base = [sys.executable, "-m", "pytest"]
        if self.config.verbose:
            base.append("-q")
        if self.config.fail_fast:
            base.append("-x")
        # Ensure junit output
        base.extend(["--junitxml", self.config.junit_xml_path])
        # Add json-report plugin if available, otherwise fall back to junit parsing later
        # Some installations have pytest-json-report; if not, create a minimal JSON from JUnit.
        json_report_tmp = self.config.json_report_path
        # Prefer plugin
        base.extend(["--json-report", f"--json-report-file={json_report_tmp}"]) if self._has_json_plugin() else None

        # Add tests list or default tests dir
        targets = tests if tests else [self.config.tests_dir]
        cmd = base + targets
        logger.info("Running pytest: %s", " ".join(cmd))

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug("pytest stdout:\n%s", proc.stdout)
        logger.debug("pytest stderr:\n%s", proc.stderr)

        if proc.returncode not in (0, 5):  # 5 is pytest no tests collected
            # We'll still try to parse reports but raise later if needed
            logger.warning("pytest returned non-zero exit code: %s", proc.returncode)

        # Prefer JSON plugin output
        if os.path.exists(json_report_tmp):
            try:
                with open(json_report_tmp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded pytest json-report from %s", json_report_tmp)
                return {"framework": "pytest", "raw": data, "rc": proc.returncode}
            except Exception:
                logger.exception("Failed to parse pytest json-report at %s", json_report_tmp)

        # Fallback: attempt to parse junit xml to a lightweight JSON
        junit_path = self.config.junit_xml_path
        if os.path.exists(junit_path):
            parsed = self._parse_junit_to_dict(junit_path)
            write_json_atomic(json_report_tmp, parsed)
            return {"framework": "pytest", "raw": parsed, "rc": proc.returncode}

        # As ultimate fallback return stdout/stderr
        return {"framework": "pytest", "raw": {"stdout": proc.stdout, "stderr": proc.stderr}, "rc": proc.returncode}

    def _has_json_plugin(self) -> bool:
        try:
            import pytest_json  # noqa: F401
            return True
        except Exception:
            # also check entry point availability by trying flag parse is complex; just return False
            return False

    def _parse_junit_to_dict(self, junit_path: str) -> Dict[str, Any]:
        # Minimal JUnit XML -> dict parser (no external deps)
        import xml.etree.ElementTree as ET
        tree = ET.parse(junit_path)
        root = tree.getroot()
        tests = []
        for ts in root.findall(".//testsuite"):
            for case in ts.findall("testcase"):
                name = case.attrib.get("name")
                classname = case.attrib.get("classname")
                time = float(case.attrib.get("time", "0"))
                status = "passed"
                message = ""
                for child in case:
                    tag = child.tag.lower()
                    if tag in ("failure", "error"):
                        status = "failed"
                        message = (child.text or "").strip()
                    elif tag == "skipped":
                        status = "skipped"
                tests.append({"name": name, "classname": classname, "time": time, "status": status, "message": message})
        summary = {
            "tests": tests,
            "counts": {
                "total": sum(1 for _ in tests),
                "passed": sum(1 for t in tests if t["status"] == "passed"),
                "failed": sum(1 for t in tests if t["status"] == "failed"),
                "skipped": sum(1 for t in tests if t["status"] == "skipped"),
            }
        }
        return summary


class UnittestAdapter(BaseAdapter):
    """
    Adapter for unittest (python stdlib). Builds a temporary runner script and
    invokes subprocess to capture output and produce a JSON-like summary.
    """

    def run(self, tests: Optional[List[str]] = None) -> Dict[str, Any]:
        runner_code = self._build_runner_script(tests)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(runner_code)
            script_path = tf.name

        cmd = [sys.executable, script_path]
        logger.info("Running unittest runner: %s %s", sys.executable, script_path)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.unlink(script_path)

        if proc.returncode != 0:
            logger.warning("unittest runner returned %s", proc.returncode)

        # Expect the runner to print a JSON blob as last line
        out = proc.stdout.strip().splitlines()
        if not out:
            return {"framework": "unittest", "raw": {"stdout": proc.stdout, "stderr": proc.stderr}, "rc": proc.returncode}

        # Try to parse last non-empty line as JSON
        for line in reversed(out):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                return {"framework": "unittest", "raw": data, "rc": proc.returncode}
            except Exception:
                break

        # Fallback
        return {"framework": "unittest", "raw": {"stdout": proc.stdout, "stderr": proc.stderr}, "rc": proc.returncode}

    def _build_runner_script(self, tests: Optional[List[str]] = None) -> str:
        """
        Build a small script that discovers and runs unittest tests, collects results,
        and prints a JSON summary to stdout (as last line).
        """
        tests_arg = tests or []
        tests_literal = repr(tests_arg)
        return f"""
import unittest, json, sys
loader = unittest.TestLoader()
suite = unittest.TestSuite()
tests = {tests_literal}
if tests:
    for t in tests:
        # If it's a file, load tests from file name module-like
        try:
            suite.addTests(loader.discover('.', pattern=t))
        except Exception:
            try:
                suite.addTests(loader.loadTestsFromName(t))
            except Exception:
                pass
else:
    suite = loader.discover('.', pattern='test_*.py')

runner = unittest.TextTestRunner(resultclass=unittest.TextTestResult)
result = runner.run(suite)

summary = {{
    "total": result.testsRun,
    "failures": len(result.failures),
    "errors": len(result.errors),
    "skipped": len(getattr(result, 'skipped', [])),
    "failures_details": [(str(t), err) for t, err in result.failures],
    "errors_details": [(str(t), err) for t, err in result.errors],
}}
print(json.dumps(summary))
"""

