"""Simple docker-compose wrapper used by tools.cli to start local stacks."""
import subprocess
from typing import Optional
from pathlib import Path


def docker_compose_up(compose_file: str, detach: bool = False) -> int:
    p = Path(compose_file)
    if not p.exists():
        print("Compose file not found:", compose_file)
        return 2

    cmd = ["docker", "compose", "-f", str(p), "up"]
    if detach:
        cmd.append("-d")
    print("+", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return 0
    except FileNotFoundError:
        print("docker compose not found. Install Docker and compose.")
        return 127
    except subprocess.CalledProcessError as e:
        print("docker compose failed:", e)
        return e.returncode
