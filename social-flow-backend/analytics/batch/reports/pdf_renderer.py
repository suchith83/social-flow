import pdfkit
import os
from .config import report_settings
from .utils import logger


class PDFRenderer:
    """Render HTML reports as PDFs"""

    def render_pdf(self, html_path: str) -> str:
        if not os.path.exists(html_path):
            raise FileNotFoundError(f"Report HTML not found: {html_path}")
        pdf_path = html_path.replace(".html", ".pdf")
        pdfkit.from_file(html_path, pdf_path)
        logger.info(f"Rendered PDF report: {pdf_path}")
        return pdf_path
