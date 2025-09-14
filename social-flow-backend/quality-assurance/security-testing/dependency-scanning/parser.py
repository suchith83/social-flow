"""
Parsers for dependency manifests (Python, Node.js, Java).
"""

import re
import toml
import json
from typing import List, Tuple
from .utils import logger


class DependencyParser:
    @staticmethod
    def parse_python(req_file: str) -> List[Tuple[str, str]]:
        """
        Parse Python requirements.txt or pyproject.toml.
        Returns list of (package, version).
        """
        deps = []
        if req_file.endswith("requirements.txt"):
            with open(req_file, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.match(r"([a-zA-Z0-9\-_]+)==([\d\.]+)", line.strip())
                    if match:
                        deps.append((match.group(1), match.group(2)))
        elif req_file.endswith("pyproject.toml"):
            data = toml.load(req_file)
            for pkg, ver in data.get("tool", {}).get("poetry", {}).get("dependencies", {}).items():
                if isinstance(ver, str):
                    deps.append((pkg, ver.strip("^")))
        logger.info(f"Parsed {len(deps)} Python dependencies")
        return deps

    @staticmethod
    def parse_node(pkg_json: str) -> List[Tuple[str, str]]:
        """
        Parse Node.js package.json.
        """
        deps = []
        with open(pkg_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            for pkg, ver in data.get("dependencies", {}).items():
                deps.append((pkg, ver.strip("^")))
        logger.info(f"Parsed {len(deps)} Node.js dependencies")
        return deps

    @staticmethod
    def parse_java(pom_file: str) -> List[Tuple[str, str]]:
        """
        Parse Java pom.xml (basic).
        """
        deps = []
        with open(pom_file, "r", encoding="utf-8") as f:
            content = f.read()
            matches = re.findall(r"<artifactId>(.*?)</artifactId>\s*<version>(.*?)</version>", content)
            deps.extend(matches)
        logger.info(f"Parsed {len(deps)} Java dependencies")
        return deps
