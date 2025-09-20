"""
Small helper to run the bundled k6 test from Python (cross-platform).

Usage:
  python runner.py --vus 50 --duration 60s --base http://localhost:8003
"""
import argparse
import shutil
import subprocess
import os
import sys

HERE = os.path.dirname(os.path.dirname(__file__))
K6_SCRIPT = os.path.join(HERE, "k6", "social_flow_test.js")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vus", type=int, default=20)
    parser.add_argument("--duration", default="30s")
    parser.add_argument("--base", default="http://localhost:8003")
    args = parser.parse_args()

    if shutil.which("k6") is None:
        print("k6 executable not found. Install k6: https://k6.io/docs/getting-started/installation/")
        sys.exit(2)

    env = os.environ.copy()
    env["K6_VUS"] = str(args.vus)
    env["K6_DURATION"] = args.duration
    env["BASE_URL"] = args.base

    cmd = ["k6", "run", K6_SCRIPT]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
