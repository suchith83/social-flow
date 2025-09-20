"""Developer helpers: find requirement files and aggregate them into a single file."""
from pathlib import Path
from typing import List
import os


def find_requirements(root: Path) -> List[Path]:
    """Recursively find runtime-requirements.txt files under root."""
    return list(root.rglob("runtime-requirements.txt"))


def aggregate_requirements(files: List[Path], out_path: Path) -> Path:
    """Aggregate multiple requirement files into out_path, deduplicating lines."""
    seen = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(f"# from: {p}\n")
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                if line not in seen:
                    seen.add(line)
                    f.write(line + "\n")
    return out_path
