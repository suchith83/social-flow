"""
Small CLI to generate fixture files for the repository.
Usage (example):
    python -m quality_assurance.testing.data.cli generate --kind users --count 100 --out tests/fixtures/users.json
"""

import argparse
import sys
from pathlib import Path
from .data_factory import DataFactory
from .config import DATA_CONFIG


def build_parser():
    p = argparse.ArgumentParser(prog="qa-data")
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate and persist fixtures")
    gen.add_argument("--kind", required=True, choices=["users", "products"])
    gen.add_argument("--count", type=int, default=DATA_CONFIG.default_batch_size)
    gen.add_argument("--out", type=str, default=None, help="Output fixture file (overrides fixtures_dir)")

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    factory = DataFactory(seed=DATA_CONFIG.default_seed)
    if args.command == "generate":
        out_path = args.out
        if out_path is None:
            filename = f"{args.kind}.json"
            out_path = str(Path(DATA_CONFIG.fixtures_dir) / filename)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        path = factory.create_and_persist(args.kind, count=args.count, filename=Path(out_path).name)
        print(f"Wrote fixtures to: {path}")


if __name__ == "__main__":
    main()
