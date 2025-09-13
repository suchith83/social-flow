# Generates validation reports (JSON/HTML)
# validation_report.py
import json
from typing import Dict, Any
from jinja2 import Template
from .utils import save_json, logger

class ValidationReport:
    """
    Aggregates validation results and exports them as JSON and HTML reports.
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def to_json(self, path: str):
        save_json(path, self.results)

    def to_html(self, path: str):
        template = Template("""
        <html>
        <head><title>Data Validation Report</title></head>
        <body>
        <h1>Data Validation Report</h1>
        <pre>{{ results | tojson(indent=2) }}</pre>
        </body>
        </html>
        """)
        html_content = template.render(results=self.results)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {path}")
