"""Simple helper to run integration tests for a targeted service."""

import os
import subprocess
import sys

def main():
    """
    Usage:
        TEST_APP_IMPORT=services.recommendation-service.src.main:app python run_integration.py
    """
    cmd = [sys.executable, "-m", "pytest", "-m", "integration", "-q"]
    env = os.environ.copy()
    print("Running integration tests:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
