"""Integration helper: run repo fixes, package common libs, aggregate requirements and run tests.

Usage:
  python scripts/integrate_and_test.py           # dry-run, shows actions
  python scripts/integrate_and_test.py --apply   # apply fixes and run steps
  python scripts/integrate_and_test.py --tests   # run pytest after apply
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_FIXER = ROOT / "scripts" / "repo_fixer.py"
COMMON_PKG = ROOT / "common" / "libraries" / "python"
AGG_REQ_OUT = ROOT / "requirements-aggregated.txt"

def run(cmd, check=True, env=None):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, env=env)

def apply_fixes(apply: bool):
    if not REPO_FIXER.exists():
        print("repo_fixer.py not found; skipping fixes")
        return
    cmd = [sys.executable, str(REPO_FIXER)]
    if apply:
        cmd.append("--apply")
    print("Running repo fixer (dry-run=%s)" % (not apply))
    run(cmd, check=True)

def install_common_editable():
    if not COMMON_PKG.exists():
        print("common package dir not found; skipping editable install")
        return
    print("Installing common library in editable mode (pip install -e)...")
    run([sys.executable, "-m", "pip", "install", "-e", str(COMMON_PKG)])

def aggregate_requirements():
    print("Aggregating runtime-requirements.txt files into", AGG_REQ_OUT)
    # simple aggregation (dedupe)
    seen = set()
    with AGG_REQ_OUT.open("w", encoding="utf-8") as out:
        out.write("# Aggregated runtime requirements\n")
        for p in ROOT.rglob("runtime-requirements.txt"):
            out.write(f"# from: {p.relative_to(ROOT)}\n")
            for line in p.read_text(encoding="utf-8").splitlines():
                l = line.strip()
                if not l or l.startswith("#") or l.startswith("//"):
                    continue
                if l not in seen:
                    seen.add(l)
                    out.write(l + "\n")
    print("Wrote aggregated requirements to", AGG_REQ_OUT)

def run_tests():
    print("Running pytest (unit & integration)...")
    run([sys.executable, "-m", "pytest", "-q"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply fixes and changes")
    parser.add_argument("--tests", action="store_true", help="Run pytest after applying")
    args = parser.parse_args()

    apply_fixes(args.apply)
    # Try to install common package editable to unify imports across services
    if args.apply:
        install_common_editable()
    aggregate_requirements()
    if args.tests and args.apply:
        run_tests()
    else:
        print("Dry-run complete. Use --apply and --tests to apply and run tests.")

if __name__ == "__main__":
    main()
