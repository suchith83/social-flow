"""
Report Generator for Container Scanning
"""

import json
import os
from jinja2 import Template
from .config import REPORT_CONFIG
from .utils import save_json, timestamp, logger


class ReportGenerator:
    def __init__(self):
        self.output_dir = REPORT_CONFIG["output_dir"]
        self.format = REPORT_CONFIG["format"]

    def generate(self, scan_results: dict) -> str:
        """Generate a report in the configured format."""
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{scan_results['image'].replace('/', '_')}_{timestamp()}.{self.format}"
        filepath = os.path.join(self.output_dir, filename)

        if self.format == "json":
            save_json(filepath, scan_results)
        elif self.format == "html":
            html_template = """
            <html>
                <head><title>Container Scan Report</title></head>
                <body>
                    <h1>Scan Report for {{ image }}</h1>
                    <p>Vulnerabilities found: {{ vulns | length }}</p>
                    <ul>
                    {% for v in vulns %}
                        <li>{{ v['id'] }} - {{ v['description'] }}</li>
                    {% endfor %}
                    </ul>
                </body>
            </html>
            """
            template = Template(html_template)
            html_content = template.render(image=scan_results["image"], vulns=scan_results["vulnerabilities"])
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

        logger.info(f"Report generated at {filepath}")
        return filepath
