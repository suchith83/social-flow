# =========================
# File: testing/security/penetration/reports/report_generator.py
# =========================
"""
Generates penetration test reports with risk scoring and evidence (non-sensitive).
- Produces JSON and HTML reports.
- Avoids logging or saving full request bodies or secrets.
"""

import json
from datetime import datetime
from ..utils.logger import get_logger

logger = get_logger("ReportGen")

def risk_score_from_findings(findings):
    """
    Simplistic risk scoring: each 'True' open port or exposed file increments score with weight.
    This is intentionally conservative; real scoring would be more nuanced.
    """
    score = 0
    details = []
    # ports
    ports = findings.get("ports", {})
    for port, open_ in ports.items():
        if open_:
            score += 5
            details.append({"type": "open_port", "port": port, "weight": 5})
    # exposed files
    exposed = findings.get("exposed_files", {})
    for fname, meta in exposed.items():
        status = meta.get("status_code")
        if status and status != 404:
            score += 8
            details.append({"type": "exposed_file", "file": fname, "status": status, "weight": 8})
    # web fingerprint issues
    fingerprint = findings.get("fingerprint", {})
    server = fingerprint.get("server")
    if server and "apache" in str(server).lower():
        score += 1
        details.append({"type": "server_banner", "banner": server, "weight": 1})
    return {"score": score, "details": details}

def generate(results: dict, target_name: str, outpath="pentest_report.json", fmt="json"):
    report = {
        "target": target_name,
        "timestamp": str(datetime.utcnow()),
        "results_summary": results,
    }
    # compute risk per target if available
    risk = {}
    for k, v in results.items():
        risk[k] = risk_score_from_findings(v)
    report["risk"] = risk

    if fmt == "json":
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"JSON pentest report written to {outpath}")
        return outpath
    elif fmt == "html":
        html = "<html><body><h1>Penetration Test Report</h1>"
        html += f"<p>Target: {target_name}</p><p>Generated: {report['timestamp']}</p>"
        for k, v in results.items():
            html += f"<h2>Section: {k}</h2><pre>{json.dumps(v, indent=2)}</pre>"
        html += "<h2>Risk Summary</h2><pre>" + json.dumps(risk, indent=2) + "</pre>"
        html += "</body></html>"
        path = outpath if outpath.endswith(".html") else outpath.replace(".json", ".html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML pentest report written to {path}")
        return path
