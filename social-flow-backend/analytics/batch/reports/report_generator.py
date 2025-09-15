import snowflake.connector
import pandas as pd
import os
from jinja2 import Environment, FileSystemLoader
from .config import report_settings
from .utils import logger


class ReportGenerator:
    """Generates business reports from analytics warehouse"""

    def __init__(self):
        self.conn = snowflake.connector.connect(report_settings.SNOWFLAKE_URI)
        self.env = Environment(loader=FileSystemLoader("analytics/batch/reports/templates"))

    def fetch_data(self, query: str) -> pd.DataFrame:
        """Execute query and return as DataFrame"""
        cur = self.conn.cursor()
        cur.execute(query)
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        cur.close()
        return df

    def generate_report(self, template_name: str, query: str, output_file: str):
        """Run query, render HTML template with data, save report"""
        df = self.fetch_data(query)
        template = self.env.get_template(template_name)
        html = template.render(data=df.to_dict(orient="records"))
        os.makedirs(report_settings.REPORT_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(report_settings.REPORT_OUTPUT_DIR, output_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Generated report: {path}")
        return path
