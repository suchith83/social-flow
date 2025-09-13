# CLI for running analytics tasks
"""
CLI utilities for running common admin tasks: process-pending, run-retention, export-range.
"""

import argparse
from .tasks import process_events_task
from .retention import run_retention_cleanup
from .export import export_aggregated_metrics, export_raw_events
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("process-now")
    sub.add_parser("run-retention")

    exp = sub.add_parser("export-range")
    exp.add_argument("start")
    exp.add_argument("end")
    args = parser.parse_args()

    if args.cmd == "process-now":
        r = process_events_task.delay()
        print("Triggered process task:", r.id)
    elif args.cmd == "run-retention":
        res = run_retention_cleanup()
        print("Retention cleanup:", res)
    elif args.cmd == "export-range":
        s = datetime.fromisoformat(args.start)
        e = datetime.fromisoformat(args.end)
        path1 = export_aggregated_metrics(s, e)
        path2 = export_raw_events(s, e)
        print("Exports saved:", path1, path2)

if __name__ == "__main__":
    main()
