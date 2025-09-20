"""
Create a development virtualenv and optionally install aggregated runtime requirements.

Usage:
  python scripts/setup_dev_env.py --create-venv --apply
  python scripts/setup_dev_env.py --dry-run
"""
from pathlib import Path
import argparse
import tempfile
import os
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def find_requirements(root: Path):
    return list(root.rglob("runtime-requirements.txt"))


def aggregate_requirements(files, out_path: Path):
    seen = set()
    with out_path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(f"# from: {p.relative_to(ROOT)}\n")
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                if line not in seen:
                    seen.add(line)
                    f.write(line + "\n")
    return out_path


def create_venv(venv_path: Path):
    import venv

    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(venv_path))
    return venv_path


def run_pip_install(venv_path: Path, requirements_file: Path):
    if os.name == "nt":
        py = venv_path / "Scripts" / "python.exe"
    else:
        py = venv_path / "bin" / "python"
    cmd = [str(py), "-m", "pip", "install", "-r", str(requirements_file)]
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-venv", action="store_true")
    parser.add_argument("--venv-path", default=str(Path(".").resolve() / ".venv"))
    parser.add_argument("--apply", action="store_true", help="perform actions (create venv / install)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    req_files = find_requirements(ROOT)
    print(f"Found {len(req_files)} requirement files.")
    for p in req_files:
        print(" -", p.relative_to(ROOT))

    with tempfile.TemporaryDirectory() as td:
        agg = Path(td) / "requirements-aggregated.txt"
        aggregate_requirements(req_files, agg)
        print("Aggregated requirements written to:", agg)
        if args.dry_run or not args.apply:
            print("DRY RUN: no changes will be applied. Pass --apply to install.")
        if args.create_venv:
            venv_path = Path(args.venv_path)
            if venv_path.exists():
                print("Virtualenv path exists:", venv_path)
            else:
                print("Will create virtualenv at:", venv_path)
                if args.apply:
                    print("Creating virtualenv...")
                    create_venv(venv_path)
                    print("Virtualenv created.")
            if args.apply:
                print("Installing aggregated requirements into venv...")
                run_pip_install(venv_path, agg)
                print("Install complete.")
            else:
                print("To install run with --apply")
        else:
            print("To create virtualenv pass --create-venv")


if __name__ == "__main__":
    main()
