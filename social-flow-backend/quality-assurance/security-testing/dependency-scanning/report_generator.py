"""
Dependency Scan Report Generator
"""

import os
import json
from jinja2 import Template
from .config import REPORT_CONFIG
from .utils import save_json, timestamp, logger


class DependencyReportGenerator:
    def __init__(self):
        self.output_dir = REPORT_CONFIG["output_dir"]
        self.format = REPORT_CONFIG["format"]

    def generate(self, scan_results: dict) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"depscan_{timestamp()}.{self.format}"
        filepath = os.path.join(self.output_dir, filename)

        if self.format == "json":
            save_json(filepath, scan_results)
        elif self.format == "html":
            template = Template("""
            <html>
            <head><title>Dependency Scan Report</title></head>
            <body>
                <h1>Dependency Scan Report</h1>
                <p>Ecosystem: {{ results.ecosystem }}</p>
                <ul>
                {% for dep in results.dependencies %}
                    <li>{{ dep.package }}@{{ dep.version }} - {{ dep.vulnerabilities | length }} vulns</li>
                {% endfor %}
                </ul>
            </body>
            </html>
            """)
            html = template.render(results=scan_results)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)

        logger.info(f"Dependency report generated at {filepath}")
        return filepath
