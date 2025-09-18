# =========================
# File: testing/security/compliance/reports/report_generator.py
# =========================
"""
Generates compliance reports in JSON, HTML, or PDF format.
"""

import json
from datetime import datetime

def generate(results, path, fmt="json"):
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": str(datetime.utcnow()), "results": results}, f, indent=2)
    elif fmt == "html":
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html><body><h1>Compliance Report</h1>")
            f.write("<p>Generated: {}</p>".format(datetime.utcnow()))
            for std, checks in results.items():
                f.write(f"<h2>{std.upper()}</h2><ul>")
                for check in checks:
                    status = "✅ PASS" if check["passed"] else "❌ FAIL"
                    f.write(f"<li>{check['policy']} ({check['severity']}): {status}</li>")
                f.write("</ul>")
            f.write("</body></html>")
    elif fmt == "pdf":
        # Simplified: in real use, integrate ReportLab or WeasyPrint
        with open(path, "w", encoding="utf-8") as f:
            f.write("PDF Compliance Report (placeholder)\n")
            f.write(json.dumps(results, indent=2))
