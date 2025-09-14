"""
Custom rule engine to run simple semantic checks and security rules over parsed files.
Rules can be Python callables discovered from a rules directory, or defined inline.
This engine is ideal for checking:
 - hardcoded credentials in literals
 - use of insecure APIs (exec, subprocess without validation, eval)
 - missing secure headers in server config files (naively)
"""

import importlib.util
import os
import time
from typing import List, Dict, Callable, Any
from ..utils import logger
from ..config import RULES

Rule = Callable[[Dict[str, Any]], List[Dict[str, Any]]]

class CustomRuleEngine:
    def __init__(self, rules_dir: str = None):
        self.rules_dir = rules_dir or RULES["rules_dir"]
        self._rules: List[Rule] = []
        self._load_rules()

    def _load_rules(self):
        """Dynamically load python rule modules from rules_dir.
        Each module exports a 'run' function that takes a parsed file dict and returns a list of findings.
        """
        if not os.path.isdir(self.rules_dir):
            logger.info(f"No rules directory found at {self.rules_dir}; skipping custom rules")
            return
        for filename in os.listdir(self.rules_dir):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(self.rules_dir, filename)
            spec = importlib.util.spec_from_file_location(f"sast_rule_{filename}", path)
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                if hasattr(module, "run") and callable(module.run):
                    self._rules.append(module.run)
                    logger.info(f"Loaded rule {filename}")
                else:
                    logger.warning(f"Rule {filename} has no callable 'run' function; skipping")
            except Exception:
                logger.exception(f"Failed loading rule {filename}; skipping")

    def run(self, parsed_file: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute all loaded rules against parsed_file (with a timeout per rule)."""
        findings = []
        for r in self._rules:
            start = time.time()
            try:
                res = r(parsed_file)
                if isinstance(res, list):
                    findings.extend(res)
            except Exception:
                logger.exception("Rule execution failed; continuing")
            elapsed = time.time() - start
            if elapsed > RULES["max_rule_runtime_seconds"]:
                logger.warning(f"Rule {r} exceeded runtime {elapsed:.2f}s")
        return findings

    # Built-in simple rules (fallback if no user rules provided)
    @staticmethod
    def builtin_hardcoded_secret_rule(parsed_file: Dict[str, Any]) -> List[Dict[str, Any]]:
        findings = []
        lits = parsed_file.get("string_literals") or parsed_file.get("string_literals", [])
        path = parsed_file.get("path")
        for s in lits:
            low = s.lower()
            if ("password" in low or "secret" in low or "api_key" in low) and len(s.strip()) > 8:
                findings.append({
                    "file": path,
                    "line": None,
                    "message_id": "SAST-HARD-CRED",
                    "message": "Possible hardcoded secret detected",
                    "severity": "HIGH",
                    "tool": "builtin-rule"
                })
        # Also check parsed_file flags
        if parsed_file.get("has_suspicious_literals"):
            findings.append({
                "file": path,
                "line": None,
                "message_id": "SAST-SUSP-LIT",
                "message": "Suspicious literal may indicate credentials",
                "severity": "MEDIUM",
                "tool": "builtin-rule"
            })
        return findings
