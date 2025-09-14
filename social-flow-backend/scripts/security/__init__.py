# scripts/security/__init__.py
"""
Security package for Social Flow

Provides:
 - Config-driven scanners and checks
 - Dependency scanning (safety, pip-audit, npm audit style wrappers)
 - Container image scanning (trivy / clair wrappers)
 - Static analysis orchestration (bandit, eslint, gosec adapters)
 - Dynamic scanning (OWASP ZAP, simple HTTP fuzzing harness)
 - Secrets detection (trufflehog-like heuristics)
 - IAM audit helpers (AWS IAM policy checks)
 - Report generation (JSON, JUnit-ish, Slack)
 - Security runner to orchestrate all tasks and integrate with CI

Design goals:
 - Quiet failures where safe; explicit failures where necessary (CI)
 - Minimal OS assumptions; make external scanners optional but automatically used if present
 - Safe defaults, timeouts, and rate-limits
"""
__version__ = "1.0.0"
__author__ = "Social Flow Security Team"
