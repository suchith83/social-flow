"""
Lightweight quality checks runner.

Runs:
- black --check
- isort --check-only
- flake8
- pytest (unit tests) and coverage report

This script is conservative: it will skip tools not installed and print guidance.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

def run_cmd(cmd, fail_ok=False):
    print(f"$ {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {exc}")
        return exc.returncode
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return 127

def main():
    rc = 0

    # black
    if shutil.which("black"):
        rc |= run_cmd(["black", "--check", "."])
    else:
        print("black not installed. Install with: pip install black")

    # isort
    if shutil.which("isort"):
        rc |= run_cmd(["isort", "--check-only", "."])
    else:
        print("isort not installed. Install with: pip install isort")

    # flake8
    if shutil.which("flake8"):
        rc |= run_cmd(["flake8", "."])
    else:
        print("flake8 not installed. Install with: pip install flake8")

    # pytest + coverage
    if shutil.which("pytest"):
        if shutil.which("coverage"):
            rc |= run_cmd(["coverage", "run", "-m", "pytest", "-q"])
            # show report
            run_cmd(["coverage", "report", "-m"])
        else:
            print("coverage not installed. Running pytest without coverage.")
            rc |= run_cmd(["pytest", "-q"])
    else:
        print("pytest not installed. Install with: pip install pytest")

    if rc != 0:
        print("One or more quality checks failed.")
    else:
        print("All quality checks passed (or were skipped due to missing tooling).")
    sys.exit(1 if rc != 0 else 0)

if __name__ == "__main__":
    main()
