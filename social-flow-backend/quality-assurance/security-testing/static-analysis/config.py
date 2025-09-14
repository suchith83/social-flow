"""
Configuration for static analysis package.
Keep environment-sensitive values in env vars or a secret manager in production.
"""

import os

# Which languages to scan by default
DEFAULT_LANGUAGES = ["python", "javascript", "java"]

# Paths and limits
SCAN_CONFIG = {
    "base_dir": os.getenv("PROJECT_ROOT", "."),    # project root to scan
    "max_file_size_kb": int(os.getenv("MAX_FILE_SIZE_KB", "512")),  # skip huge files
    "exclude_paths": ["node_modules", "venv", ".git", "build", "dist"],
    "concurrency": int(os.getenv("SAST_CONCURRENCY", "4")),
}

# External tool configuration (if present on system)
TOOLS = {
    "pylint_cmd": os.getenv("PYLINT_CMD", "pylint"),
    "eslint_cmd": os.getenv("ESLINT_CMD", "eslint"),
    "java_static_tool": os.getenv("JAVA_STATIC_TOOL", "spotbugs")  # placeholder
}

# Reporting config
REPORT_CONFIG = {
    "output_dir": os.getenv("SAST_REPORT_DIR", "./sast-reports"),
    "formats": ["json", "html"],  # supported outputs
    "severity_threshold": os.getenv("SAST_SEVERITY_THRESHOLD", "LOW"),  # LOW/MEDIUM/HIGH/CRITICAL
}

# Custom rule engine settings (for rules in rules/)
RULES = {
    "rules_dir": os.getenv("SAST_RULES_DIR", "./sast_rules"),
    "max_rule_runtime_seconds": int(os.getenv("SAST_RULE_TIMEOUT", "5"))
}
