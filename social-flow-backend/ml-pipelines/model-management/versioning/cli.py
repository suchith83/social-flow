# CLI for registry operations
# cli.py
"""
Tiny CLI to interact with the ModelRegistry.
Usage:
  python -m ml_pipelines.model_management.versioning.cli register --name model --file path/to/model.pkl
  python -m ml_pipelines.model_management.versioning.cli list --name model
  python -m ml_pipelines.model_management.versioning.cli fetch --name model --version 1.0.0 --out out.pkl
  python -m ml_pipelines.model_management.versioning.cli promote --name model --version 1.0.0 --tag production
  python -m ml_pipelines.model_management.versioning.cli gc --name model
"""

import argparse
import json
import logging
from registry import ModelRegistry
from metadata_store import SQLiteMetadataStore
from storage_adapter import LocalStorageAdapter
from retention_policy import RetentionPolicy
from utils import now_iso

logging.basicConfig(level=logging.INFO)


def build_parser():
    p = argparse.ArgumentParser(prog="model-registry")
    sub = p.add_subparsers(dest="cmd")

    reg = sub.add_parser("register")
    reg.add_argument("--name", required=True)
    reg.add_argument("--file", required=True, help="Path to artifact to register")
    reg.add_argument("--version", required=False)
    reg.add_argument("--created-by", required=False)
    reg.add_argument("--meta", required=False, help="JSON metadata string")
    reg.add_argument("--prov", required=False, help="JSON provenance string")
    reg.add_argument("--auto-bump", choices=["major", "minor", "patch"], default="patch")

    lst = sub.add_parser("list")
    lst.add_argument("--name", required=True)

    latest = sub.add_parser("latest")
    latest.add_argument("--name", required=True)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--name", required=True)
    fetch.add_argument("--version", required=True)
    fetch.add_argument("--out", required=True)

    promote = sub.add_parser("promote")
    promote.add_argument("--name", required=True)
    promote.add_argument("--version", required=True)
    promote.add_argument("--tag", required=True)

    delete = sub.add_parser("delete")
    delete.add_argument("--name", required=True)
    delete.add_argument("--version", required=True)

    gc = sub.add_parser("gc")
    gc.add_argument("--name", required=True)
    gc.add_argument("--keep", type=int, default=5)

    return p


def main():
    p = build_parser()
    args = p.parse_args()

    store = SQLiteMetadataStore()
    storage = LocalStorageAdapter()
    reg = ModelRegistry(store=store, storage=storage)

    if args.cmd == "register":
        meta = json.loads(args.meta) if args.meta else {}
        prov = json.loads(args.prov) if args.prov else {}
        art = reg.register_model(name=args.name, src_path=args.file, version=args.version, created_by=args.created_by, metadata=meta, provenance=prov, auto_bump=args.auto_bump)
        print(art.to_json())
    elif args.cmd == "list":
        arts = reg.list_versions(args.name)
        for a in arts:
            print(a.to_json())
    elif args.cmd == "latest":
        a = reg.latest(args.name)
        if a:
            print(a.to_json())
        else:
            print("{}")
    elif args.cmd == "fetch":
        local = reg.fetch_model(args.name, args.version, args.out)
        print("Downloaded to", local)
    elif args.cmd == "promote":
        reg.promote(args.name, args.version, args.tag)
        print("Promoted")
    elif args.cmd == "delete":
        reg.delete(args.name, args.version)
        print("Deleted")
    elif args.cmd == "gc":
        rp = RetentionPolicy(store=store, storage=storage, keep_latest=args.keep)
        deleted = rp.collect(args.name)
        for d in deleted:
            print("Deleted:", d.name, d.version)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
