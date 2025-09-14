"""
Command-line interface for the static analysis package.

Example:
    python -m static_analysis.cli --root /path/to/project --languages python,javascript --severity HIGH
"""

import argparse
import sys
from .orchestrator import StaticAnalysisOrchestrator
from .config import SCAN_CONFIG
from .utils import logger

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Static Application Security Testing (SAST)")
    parser.add_argument("--root", "-r", default=SCAN_CONFIG["base_dir"], help="Project root to scan")
    parser.add_argument("--languages", "-l", default=",".join(StaticAnalysisOrchestrator().languages),
                        help="Comma-separated languages to scan (python,javascript,java)")
    parser.add_argument("--severity", "-s", default=None, help="Minimum severity to include (LOW/MEDIUM/HIGH/CRITICAL)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable multiprocessing for debugging")
    args = parser.parse_args(argv)

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    orchestrator = StaticAnalysisOrchestrator(languages=langs)
    if args.no_parallel:
        # monkey-patch multiprocessing to single-threaded: not necessary here but kept as toggle
        logger.info("Running in single-process mode (no-parallel)")
        orchestrator.config["concurrency"] = 1

    res = orchestrator.run(args.root, severity_threshold=args.severity)
    logger.info("SAST complete. Reports: %s", res["reports"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
