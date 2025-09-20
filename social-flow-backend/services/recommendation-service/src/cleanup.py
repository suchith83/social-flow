"""
Cleanup helper: stop docker compose stack (deployment/docker-compose.dev.yml)
and remove local temp artifacts (.data/dev.db, .certs).
Usage:
  python scripts/cleanup.py --stop-compose --remove-data
"""
import argparse
import subprocess
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "deployment" / "docker-compose.dev.yml"
DB_FILE = ROOT / ".data" / "dev.db"
CERT_DIR = ROOT / ".certs"


def run(cmd):
    print("+", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print("Command failed:", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-compose", action="store_true")
    parser.add_argument("--remove-data", action="store_true")
    parser.add_argument("--remove-certs", action="store_true")
    args = parser.parse_args()

    if args.stop_compose:
        if COMPOSE_FILE.exists():
            print("Stopping docker-compose stack...")
            run(["docker", "compose", "-f", str(COMPOSE_FILE), "down"])
        else:
            print("Compose file not found; skipping stop.")

    if args.remove_data:
        if DB_FILE.exists():
            print("Removing DB:", DB_FILE)
            DB_FILE.unlink()
        else:
            print("No DB file found:", DB_FILE)

    if args.remove_certs:
        if CERT_DIR.exists():
            print("Removing certs dir:", CERT_DIR)
            for p in CERT_DIR.glob("*"):
                p.unlink()
            CERT_DIR.rmdir()
        else:
            print("No certs dir found:", CERT_DIR)


if __name__ == "__main__":
    main()
