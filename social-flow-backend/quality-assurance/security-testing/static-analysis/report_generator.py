"""
Consolidates findings from multiple analyzers and generates reports in JSON and HTML.
Reports include metadata, counts, severity summaries, and per-file details.
"""

import os
import json
from jinja2 import Template
from .utils import save_json, timestamp, logger, SEVERITY_ORDER
from .config import REPORT_CONFIG

class StaticReportGenerator:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or REPORT_CONFIG["output_dir"]
        self.formats = REPORT_CONFIG["formats"]

    def _summarize(self, findings: list) -> dict:
        summary = {"total": len(findings), "by_severity": {}, "by_tool": {}}
        for f in findings:
            sev = f.get("severity", "LOW")
            summary["by_severity"].setdefault(sev, 0)
            summary["by_severity"][sev] += 1
            tool = f.get("tool", "unknown")
            summary["by_tool"].setdefault(tool, 0)
            summary["by_tool"][tool] += 1
        return summary

    def generate(self, findings: list, meta: dict = None) -> dict:
        """
        Create reports in configured formats. Returns a dict with paths for generated files.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        meta = meta or {}
        report_base = f"sast_report_{timestamp()}"
        result = {"generated": {}, "summary": None}

        summary = self._summarize(findings)
        result["summary"] = summary
        payload = {
            "metadata": meta,
            "summary": summary,
            "findings": findings
        }

        if "json" in self.formats:
            path = os.path.join(self.output_dir, report_base + ".json")
            save_json(path, payload)
            result["generated"]["json"] = path

        if "html" in self.formats:
            path = os.path.join(self.output_dir, report_base + ".html")
            html = self._render_html(payload)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            result["generated"]["html"] = path

        logger.info(f"Generated reports: {result['generated']}")
        return result

    def _render_html(self, payload: dict) -> str:
        template = Template("""
        <html>
        <head><title>SAST Report</title></head>
        <body>
            <h1>SAST Report</h1>
            <p>Generated: {{ metadata.generated_at }}</p>
            <h2>Summary</h2>
            <ul>
            {% for sev, count in summary.by_severity.items() %}
                <li>{{ sev }}: {{ count }}</li>
            {% endfor %}
            </ul>
            <h2>By Tool</h2>
            <ul>
            {% for tool, count in summary.by_tool.items() %}
                <li>{{ tool }}: {{ count }}</li>
            {% endfor %}
            </ul>
            <h2>Findings</h2>
            <table border="1" cellpadding="6">
                <thead><tr><th>File</th><th>Line</th><th>Message</th><th>Severity</th><th>Tool</th></tr></thead>
                <tbody>
                {% for f in findings %}
                <tr>
                    <td>{{ f.file }}</td>
                    <td>{{ f.line or "-" }}</td>
                    <td>{{ f.message }}</td>
                    <td>{{ f.severity }}</td>
                    <td>{{ f.tool }}</td>
                </tr>
                {% endfor %}
                </tbody>
            </table>
        </body>
        </html>
        """)
        meta = payload.get("metadata", {})
        meta.setdefault("generated_at", timestamp())
        return template.render(metadata=meta, summary=payload["summary"], findings=payload["findings"])
