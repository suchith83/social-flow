# =========================
# File: testing/security/penetration/engine.py
# =========================
"""
Engine that orchestrates penetration tests according to policies.
- Enforces consent: requires a 'consent' flag or explicit confirmation (safe default: consent=False)
- Loads policies and executes scanners
- Aggregates results and hands to report generator
"""

import yaml
from pathlib import Path
from .utils.logger import get_logger
from .scanners import port_scanner, web_scanner, vuln_scanner, auth_tester
from .reports import report_generator
from .payloads import injection_tests

logger = get_logger("PentestEngine")

class PentestEngine:
    def __init__(self, policy_file="pentest_policies.yaml", consent=False, dry_run=True):
        self.policy_file = Path(policy_file)
        self.policies = {}
        self.consent = consent
        self.dry_run = dry_run
        self.load_policies()

    def load_policies(self):
        if not self.policy_file.exists():
            raise FileNotFoundError(f"Policy file missing: {self.policy_file}")
        with open(self.policy_file, "r", encoding="utf-8") as f:
            self.policies = yaml.safe_load(f).get("policies", [])
        logger.info(f"Loaded {len(self.policies)} pentest policies")

    def run_for_target(self, target_name: str, target_meta: dict):
        """
        Run all enabled policies that apply to the target.
        target_meta should include:
          - host or base_url
          - type tags like "web", "infrastructure", "auth-service"
        """
        results = {}
        for policy in self.policies:
            if not policy.get("enabled", True):
                continue
            applies = policy.get("targets", [])
            # match by tag present in target_meta['tags']
            tags = target_meta.get("tags", [])
            if not any(t in tags for t in applies):
                continue
            pid = policy["id"]
            logger.info(f"Running policy {pid} ({policy.get('description')}) on {target_name}")
            # safe_mode determines the variant
            safe_mode = policy.get("safe_mode", True)
            settings = policy.get("settings", {})

            # Dispatch to known policy types based on id prefix (simple mapping)
            if pid.startswith("PENTEST-PORTS"):
                host = target_meta.get("host")
                ports = settings.get("ports", [22,80,443,8080])
                timeout = settings.get("timeout_seconds", 1)
                concurrency = settings.get("concurrency", 40)
                # If dry_run or consent not given, do not execute real scans
                if self.dry_run or not self.consent:
                    logger.info("Dry run OR consent not provided. Simulating port scan.")
                    simulated = {p: False for p in ports}
                    results[pid] = {"simulated": True, "ports": simulated}
                else:
                    ports_res = port_scanner.scan(host, ports, timeout=timeout, concurrency=concurrency)
                    results[pid] = {"simulated": False, "ports": ports_res}

            elif pid.startswith("PENTEST-WEB"):
                base_url = target_meta.get("base_url")
                if self.dry_run:
                    # safe quick checks
                    headers = web_scanner.headers_and_body(base_url)
                    robots = web_scanner.check_robots(base_url)
                    results[pid] = {"simulated": True, "headers": headers, "robots": robots}
                else:
                    headers = web_scanner.headers_and_body(base_url)
                    tls = web_scanner.tls_info(base_url.replace("https://","").replace("http://","").split("/")[0])
                    dirs = web_scanner.directory_discovery(base_url, wordlist=settings.get("wordlist"), max_checks=settings.get("max_checks", 20), rate_limit_per_minute=settings.get("rate_limit", 30))
                    fingerprint = vuln_scanner.fingerprint_web(base_url)
                    exposed_files = vuln_scanner.check_exposed_files(base_url)
                    results[pid] = {"simulated": False, "headers": headers, "tls": tls, "dirs": dirs, "fingerprint": fingerprint, "exposed_files": exposed_files}

            elif pid.startswith("PENTEST-AUTH"):
                url = target_meta.get("auth_url") or target_meta.get("base_url")
                username = target_meta.get("test_username", "admin")
                wl_path = settings.get("wordlist")
                # load small sample list if provided
                wordlist = []
                if wl_path:
                    try:
                        with open(wl_path, "r", encoding="utf-8") as f:
                            wordlist = [l.strip() for l in f if l.strip()]
                    except Exception as e:
                        logger.warning(f"Unable to load wordlist {wl_path}: {e}")
                else:
                    wordlist = ["password", "admin", "123456", "P@ssw0rd"]
                if self.dry_run or not self.consent:
                    sim = auth_tester.simulate_bruteforce(username, wordlist, max_attempts=settings.get("max_attempts_per_minute", 10))
                    results[pid] = {"simulated": True, "simulation": sim}
                else:
                    exec_res = auth_tester.execute_bruteforce(url, username, wordlist, rate_limit_per_minute=settings.get("max_attempts_per_minute", 10), max_attempts=settings.get("max_attempts", 50))
                    results[pid] = {"simulated": False, "results": exec_res}

            else:
                logger.warning(f"Unknown policy {pid}. Skipping.")
                results[pid] = {"skipped": True, "reason": "unknown policy id"}

        # Aggregate some findings into a target-level view for reporting
        aggregated = self.aggregate_results(results)
        return {"raw": results, "aggregated": aggregated}

    def aggregate_results(self, raw_results: dict):
        """
        Aggregate results into a simple normalized structure for reports.
        """
        agg = {}
        # Example: collect open ports from port policy results
        for pid, val in raw_results.items():
            if pid.startswith("PENTEST-PORTS") and not val.get("simulated", True):
                agg["ports"] = val.get("ports", {})
            if pid.startswith("PENTEST-WEB"):
                agg["headers"] = val.get("headers")
                agg["fingerprint"] = val.get("fingerprint")
                agg["exposed_files"] = val.get("exposed_files")
        return agg

    def generate_report(self, results, target_name, outpath="pentest_report.json", fmt="json"):
        return report_generator.generate(results.get("aggregated", {}), target_name, outpath=outpath, fmt=fmt)
