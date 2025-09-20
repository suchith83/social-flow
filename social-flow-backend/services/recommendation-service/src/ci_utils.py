"""
Simple CI utility functions callable from CI jobs or locally.

Usage:
  python -m automation.ci_utils run-tests --path services/recommendation-service
  python -m automation.ci_utils lint --path services/recommendation-service
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_pytest(path: Path) -> int:
    print(f"Running pytest in: {path}")
    cmd = [sys.executable, "-m", "pytest", str(path)]
    return subprocess.call(cmd)


def run_flake8(path: Path) -> int:
    print(f"Running flake8 in: {path}")
    cmd = [sys.executable, "-m", "flake8", str(path)]
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("run-tests")
    t.add_argument("--path", default=".", help="Path to run pytest against")

    l = sub.add_parser("lint")
    l.add_argument("--path", default=".", help="Path to lint with flake8")

    args = parser.parse_args()
    if args.cmd == "run-tests":
        rc = run_pytest(Path(args.path))
        sys.exit(rc)
    if args.cmd == "lint":
        rc = run_flake8(Path(args.path))
        sys.exit(rc)
    parser.print_help()


if __name__ == "__main__":
    main()
