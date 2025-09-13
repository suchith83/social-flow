# Helpers: templating, exec, logging, config
# ================================================================
# File: utils.py
# Purpose: Shared utilities (templating, shell exec, logging, config)
# ================================================================

import subprocess
import logging
import yaml
import os
from typing import Dict, Any
from pathlib import Path
import tempfile
import json
import time

DEFAULT_LOG_LEVEL = logging.INFO


def setup_logger(name: str, level: int = DEFAULT_LOG_LEVEL):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger("deployment.utils")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_file(path: str, content: str, mode: str = "w", make_dir: bool = True):
    p = Path(path)
    if make_dir:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    logger.info(f"wrote file: {path}")


def render_template(template: str, context: Dict[str, Any]) -> str:
    """Tiny, safe templating using Python format mapping."""
    return template.format(**{k: safe_primitive(v) for k, v in context.items()})


def safe_primitive(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


def run_cmd(cmd: str, cwd: str = None, capture_output: bool = True, check: bool = True, env: Dict[str, str] = None):
    logger.info(f"running command: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"command failed ({result.returncode}): {cmd}\nstdout: {result.stdout}\nstderr: {result.stderr}")
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return result.stdout.strip()


def timestamp():
    return int(time.time())


def unique_tag(base: str = "model"):
    return f"{base}-{timestamp()}"
