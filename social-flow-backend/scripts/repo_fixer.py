"""
Repository fixer for common accidental problems found during repo sweep.

Usage:
  python scripts/repo_fixer.py --dry-run         # show files that would change
  python scripts/repo_fixer.py --apply           # apply changes (backups created *.bak)
"""
from __future__ import annotations
import argparse
import os
import sys
import io
import shutil
import fnmatch
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Files/directories to ignore while scanning
IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "build", "dist", ".cache", ".idea", ".pytest_cache"}
IGNORE_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.zip", "*.tar.gz", "*.tgz", "*.pdf", "*.woff", "*.woff2", "*.exe", "*.dll"]


def should_ignore(path: str) -> bool:
    parts = path.split(os.sep)
    if any(p in IGNORE_DIRS for p in parts):
        return True
    for pat in IGNORE_PATTERNS:
        if fnmatch.fnmatch(path, pat):
            return True
    return False


def normalize_text(content: str) -> str:
    """
    Apply transformations:
    - Convert leading lines that start with '//' into '#' (useful if snippets with // filepath: were embedded).
    - Remove lines that look like injected "filepath" comment lines (e.g. starting with 'filepath:' or '// filepath:').
    - Collapse consecutive duplicate lines (simple dedupe).
    - Trim trailing whitespace on lines, but preserve file endings.
    """
    lines = content.splitlines()
    out_lines: List[str] = []
    prev_line = None
    for raw in lines:
        line = raw.rstrip()  # remove trailing whitespace
        # convert C-style comment header lines that start with // into python/text comments
        if line.startswith("//"):
            # if it's like "// filepath:" drop the line entirely (it is an injected artifact)
            if "filepath:" in line.lower():
                # skip injected filepath markers
                continue
            # otherwise convert to hash-style comment
            line = "#" + line[2:].lstrip()
        # drop injected HTML/Markdown comment lines that contain "filepath:" (e.g., "<!-- filepath: ... -->")
        if "filepath:" in line and ("<!--" in line or "filepath:" in line and line.strip().startswith("filepath:")):
            # skip artifact lines
            continue

        # Collapse consecutive duplicate lines (very conservative)
        if prev_line is not None and line == prev_line:
            # skip duplicate line
            continue

        out_lines.append(line)
        prev_line = line

    # Preserve a trailing newline
    return "\n".join(out_lines) + ("\n" if content.endswith("\n") else "")


def process_file(path: str, apply: bool = False) -> bool:
    """
    Read file, normalize, and optionally write back. Returns True if file would change / changed.
    """
    try:
        # skip binary files by attempting to decode as text
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return False  # binary file -> ignore
    except Exception:
        return False

    new_text = normalize_text(text)
    if new_text == text:
        return False

    if not apply:
        print("[DRY] Would modify:", path)
        return True

    # backup original
    bak = path + ".bak"
    try:
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
    except Exception:
        print("Warning: failed to create backup for", path)

    # write updated content atomically
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
    os.replace(tmp_path, path)
    print("Updated:", path, " (backup at: {})".format(bak))
    return True


def walk_and_fix(root: str, apply: bool = False, show_summary: bool = True) -> int:
    changed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root)
            if should_ignore(rel):
                continue
            # limit to text files by extension heuristics (common code & config)
            ext = os.path.splitext(fname)[1].lower()
            if ext in {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml", ".dockerfile", ".lock", ".txt", ".rst"} or fname.lower().startswith("runtime-requirements") or "requirements" in fname.lower():
                try:
                    if process_file(fpath, apply=apply):
                        changed += 1
                except Exception as exc:
                    print("Error processing", fpath, exc)
                    continue
    if show_summary:
        print(f"Files changed/would change: {changed}")
    return changed


def main():
    parser = argparse.ArgumentParser(description="Repository fixer for common injected artifacts and simple normalizations.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Without --apply the script runs in dry-run mode.")
    parser.add_argument("--root", default=ROOT, help="Repository root to scan.")
    args = parser.parse_args()

    print("Scanning root:", args.root)
    count = walk_and_fix(args.root, apply=args.apply)
    if args.apply:
        print("Apply completed. Review .bak files for backups.")
    else:
        print("Dry-run complete. Rerun with --apply to modify files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
