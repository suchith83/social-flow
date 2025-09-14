# Creates dashboards/reports from analysis results
# monitoring/logging/analysis/report_generator.py
"""
Report generator for log analysis results.
Exports results to JSON and HTML for dashboards.
"""

import json
from pathlib import Path
from jinja2 import Template
from .config import CONFIG
from .utils import serialize


class ReportGenerator:
    def __init__(self):
        self.output_dir = CONFIG["REPORTS"]["output_dir"]
        self.formats = CONFIG["REPORTS"]["formats"]
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def generate(self, analysis_results: dict, name="report"):
        """Generate reports in configured formats."""
        outputs = []
        if "json" in self.formats:
            path = Path(self.output_dir) / f"{name}.json"
            with open(path, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)
            outputs.append(str(path))

        if "html" in self.formats:
            template = Template("""
            <html>
              <head><title>Log Analysis Report</title></head>
              <body>
                <h1>Log Analysis Report</h1>
                <pre>{{ data }}</pre>
              </body>
            </html>
            """)
            path = Path(self.output_dir) / f"{name}.html"
            with open(path, "w") as f:
                f.write(template.render(data=serialize(analysis_results)))
            outputs.append(str(path))

        return outputs
