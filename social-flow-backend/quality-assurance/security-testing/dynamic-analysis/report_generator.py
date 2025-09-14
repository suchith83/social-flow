"""
DAST Report Generator
"""

import os
from jinja2 import Template
from .config import REPORT_CONFIG
from .utils import save_json, timestamp, logger


class DynamicReportGenerator:
    def __init__(self):
        self.output_dir = REPORT_CONFIG["output_dir"]
        self.format = REPORT_CONFIG["format"]

    def generate(self, scan_results: dict) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"dast_{timestamp()}.{self.format}"
        filepath = os.path.join(self.output_dir, filename)

        if self.format == "json":
            save_json(filepath, scan_results)
        elif self.format == "html":
            template = Template("""
            <html>
            <head><title>Dynamic Analysis Report</title></head>
            <body>
                <h1>DAST Report for {{ target }}</h1>
                <ul>
                {% for v in vulns %}
                    <li><b>{{ v.type }}</b> ({{ v.severity }}): {{ v.description }} 
                    <br> Mitigation: {{ v.mitigation }}</li>
                {% endfor %}
                </ul>
            </body>
            </html>
            """)
            html = template.render(target=scan_results["target"], vulns=scan_results["vulnerabilities"])
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)

        logger.info(f"DAST report generated at {filepath}")
        return filepath
