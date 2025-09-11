# sanitizer.py
import html
import re


class Sanitizer:
    """
    Input sanitization for preventing XSS, SQL Injection, etc.
    """

    @staticmethod
    def escape_html(data: str) -> str:
        return html.escape(data)

    @staticmethod
    def sanitize_sql(data: str) -> str:
        return re.sub(r"[;'\"]", "", data)

    @staticmethod
    def sanitize_input(data: str) -> str:
        return Sanitizer.escape_html(Sanitizer.sanitize_sql(data))
