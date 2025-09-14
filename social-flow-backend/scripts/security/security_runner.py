# scripts/security/security_runner.py
import logging
import argparse
import sys
from typing import Dict, Any

from .config_loader import ConfigLoader
from .logger import configure_logging
from .dependency_scanner import DependencyScanner
from .container_scanner import ContainerScanner
from .static_analyzer import StaticAnalyzer
from .dynamic_scanner import DynamicScanner
from .secrets_detector import SecretsDetector
from .iam_audit import IAMAudit
from .reporter import Reporter
from .utils import safe_mkdir

logger = logging.getLogger("security.runner")


def run_all(config_path: str = "security.yaml", output_dir: str = None) -> Dict[str, Any]:
    cfg = ConfigLoader(config_path).load()
    if output_dir:
        cfg["security"]["reporter"]["output_dir"] = output_dir
    # configure logging early
    log_path = cfg.get("security", {}).get("log_path", "./logs/security.log")
    configure_logging(log_path)

    # ensure output dir
    safe_mkdir(cfg["security"]["reporter"]["output_dir"])

    results = {}
    # dependency
    deps = DependencyScanner(cfg)
    try:
        results["dependency"] = deps.run()
    except Exception:
        logger.exception("Dependency scan failed")
        results["dependency"] = {"error": True}

    # static
    static = StaticAnalyzer(cfg)
    try:
        results["static"] = static.run()
    except Exception:
        logger.exception("Static analysis failed")
        results["static"] = {"error": True}

    # container
    container = ContainerScanner(cfg)
    try:
        results["container"] = container.run()
    except Exception:
        logger.exception("Container scanning failed")
        results["container"] = {"error": True}

    # secrets
    secrets = SecretsDetector(cfg)
    try:
        results["secrets"] = secrets.run()
    except Exception:
        logger.exception("Secrets detection failed")
        results["secrets"] = {"error": True}

    # dynamic
    dynamic = DynamicScanner(cfg)
    try:
        results["dynamic"] = dynamic.run()
    except Exception:
        logger.exception("Dynamic scanning failed")
        results["dynamic"] = {"error": True}

    # iam audit (optional, may require boto3 and credentials)
    iam = IAMAudit(cfg)
    try:
        results["iam"] = iam.run()
    except Exception:
        logger.exception("IAM audit failed")
        results["iam"] = {"error": True}

    # reporting
    reporter = Reporter(cfg)
    try:
        aggregated = reporter.aggregate(results)
        results["report"] = aggregated
    except Exception:
        logger.exception("Reporting failed")
        results["report"] = {"error": True}

    # CI policy enforcement - fail CI if configured severity threshold exceeded
    try:
        ci_cfg = cfg.get("security", {}).get("ci", {})
        if ci_cfg.get("fail_on_high", True):
            # naive policy: if total findings > 0 and high threshold triggered
            total = results.get("report", {}).get("summary", {}).get("total_findings", 0)
            high_threshold = int(ci_cfg.get("high_severity_threshold", 7))
            if total > 0:
                logger.warning("Security scan found %d findings", total)
                # For now, fail if any findings exist — customize as needed
                # If more complex severity parsing is needed, parse tool outputs earlier.
                sys.exit(3)
    except SystemExit:
        raise
    except Exception:
        logger.exception("CI policy enforcement failed")

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run security scans")
    parser.add_argument("--config", default="security.yaml", help="Path to config file")
    parser.add_argument("--out", default=None, help="Output directory override")
    args = parser.parse_args(argv)
    run_all(args.config, output_dir=args.out)


if __name__ == "__main__":
    main()
