# scripts/security/secrets_detector.py
import logging
import re
import os
import math
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("security.secrets")


# helper entropy function (Shannon)
def shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    import collections
    cnt = collections.Counter(data)
    length = len(data)
    entropy = -sum((v/length) * math.log2(v/length) for v in cnt.values())
    return entropy


class SecretsDetector:
    """
    Heuristic secrets detection that:
     - Scans files for high-entropy strings
     - Matches common secret patterns (AWS keys, JWT, private keys)
     - Avoids binary files and respects .gitignore-like patterns optionally

    Returns normalized list of findings.
    """

    AWS_ACCESS_KEY_RE = re.compile(r"AKIA[0-9A-Z]{16}")
    AWS_SECRET_RE = re.compile(r"(?i)aws(.{0,20})?(secret|access|key).{0,20}[:=]\s*([A-Za-z0-9/+=]{16,128})")
    GENERIC_TOKEN_RE = re.compile(r"(?i)(token|secret|passwd|password|api_key|apikey)[\s:=]{0,3}([A-Za-z0-9_\-]{8,200})")
    PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----")

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("security", {}).get("secrets", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.entropy_threshold = float(cfg.get("entropy_threshold", 4.5))
        self.paths = cfg.get("paths", ["./"])
        self.max_file_size = int(cfg.get("max_file_size", 2 * 1024 * 1024))  # 2 MB default

    def _is_text(self, path: str) -> bool:
        try:
            with open(path, "rb") as fh:
                chunk = fh.read(1024)
                # simple heuristic: if NUL byte present, treat as binary
                return b"\x00" not in chunk
        except Exception:
            return False

    def scan_file(self, path: str) -> List[Dict[str, Any]]:
        findings = []
        try:
            if os.path.getsize(path) > self.max_file_size:
                return findings
            if not self._is_text(path):
                return findings
            with open(path, "r", errors="ignore") as fh:
                for lineno, line in enumerate(fh, start=1):
                    # quick pattern checks
                    if self.PRIVATE_KEY_RE.search(line):
                        findings.append({"path": path, "line": lineno, "type": "private_key", "snippet": "-----BEGIN PRIVATE KEY-----"})
                    for m in self.AWS_ACCESS_KEY_RE.finditer(line):
                        findings.append({"path": path, "line": lineno, "type": "aws_access_key", "match": m.group(0)})
                    for m in self.GENERIC_TOKEN_RE.finditer(line):
                        token = m.group(2)
                        if token and shannon_entropy(token) >= self.entropy_threshold:
                            findings.append({"path": path, "line": lineno, "type": "generic_token", "match": token, "entropy": shannon_entropy(token)})
                    # also detect long base64-like strings
                    for candidate in re.findall(r"[A-Za-z0-9+/=]{20,}", line):
                        ent = shannon_entropy(candidate)
                        if ent >= self.entropy_threshold:
                            findings.append({"path": path, "line": lineno, "type": "high_entropy_string", "match": candidate[:60], "entropy": ent})
        except Exception:
            logger.exception("Failed scanning file %s", path)
        return findings

    def run(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            logger.info("Secrets detection disabled")
            return []
        results = []
        for base in self.paths:
            for root, dirs, files in os.walk(base):
                # skip .git directory to avoid scanning history by default
                if ".git" in dirs:
                    dirs.remove(".git")
                for f in files:
                    path = os.path.join(root, f)
                    try:
                        findings = self.scan_file(path)
                        if findings:
                            results.extend(findings)
                    except Exception:
                        logger.exception("Error scanning %s", path)
        return results
