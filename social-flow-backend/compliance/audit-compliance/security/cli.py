# cli.py
"""
Tiny CLI to run key security compliance workflows.
This is intentionally minimal — integrate with orchestration/orchestration (Airflow/Argo/CI) in production.
"""
import argparse
import json
from .vulnerability_scanner import VulnerabilityScanner
from .incident_response import IncidentResponseEngine
from .secrets_management import FileSecretsAdapter, SecretsManager
from .utils import log_event
import datetime

def run_scan(args):
    # load inventory JSON
    with open(args.inventory, "r", encoding="utf-8") as f:
        inventory = json.load(f)
    # load feed if provided
    feed = []
    if args.feed:
        with open(args.feed, "r", encoding="utf-8") as f:
            feed = json.load(f)
    # convert disclosed_on strings to dates if present
    for v in feed:
        if isinstance(v.get("disclosed_on"), str):
            v["disclosed_on"] = datetime.date.fromisoformat(v["disclosed_on"])
    scanner = VulnerabilityScanner(feed)
    findings = scanner.scan_inventory(inventory)
    print(json.dumps(findings, indent=2))
    log_event("Scan completed", level="INFO", findings_count=len(findings))

def run_discover(args):
    adapter = FileSecretsAdapter(path=args.store)
    manager = SecretsManager(adapter)
    findings = manager.discover_in_files(args.path)
    print(json.dumps(findings, indent=2))
    log_event("Discovery completed", level="INFO", path=args.path, findings=len(findings))

def main():
    parser = argparse.ArgumentParser("security_audit")
    sub = parser.add_subparsers(dest="cmd")
    scan = sub.add_parser("scan", help="Run vulnerability scan")
    scan.add_argument("--inventory", required=True, help="Inventory JSON file")
    scan.add_argument("--feed", required=False, help="Vuln feed JSON file")
    discover = sub.add_parser("discover", help="Discover secrets in repo")
    discover.add_argument("--path", required=True, help="Root path to scan")
    discover.add_argument("--store", default="secrets.json", help="File-backed secret store")
    args = parser.parse_args()
    if args.cmd == "scan":
        run_scan(args)
    elif args.cmd == "discover":
        run_discover(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
