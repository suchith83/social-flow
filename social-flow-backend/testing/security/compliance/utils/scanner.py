# =========================
# File: testing/security/compliance/utils/scanner.py
# =========================
"""
Scanner utilities to check files, endpoints, logs for compliance evidence.
"""

import os
import re
import ssl
import socket

def scan_files_for_keywords(keywords, directory):
    findings = {}
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                with open(os.path.join(root, file), "r", errors="ignore") as f:
                    content = f.read()
                    findings[file] = any(k.lower() in content.lower() for k in keywords)
            except Exception:
                findings[file] = False
    return findings

def check_encryption(config_file):
    try:
        with open(config_file, "r") as f:
            return {"encrypted": "AES" in f.read() or "TLS" in f.read()}
    except FileNotFoundError:
        return {"encrypted": False}

def scan_endpoints_for_tls(url):
    # Basic TLS handshake check
    try:
        hostname = url.split("//")[-1].split("/")[0]
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                return {"tls": ssock.version() is not None}
    except Exception:
        return {"tls": False}

def scan_logs_for_pattern(logfile, pattern):
    try:
        with open(logfile, "r") as f:
            content = f.read()
            return {"contains_errors": bool(re.search(pattern, content))}
    except FileNotFoundError:
        return {"contains_errors": False}
