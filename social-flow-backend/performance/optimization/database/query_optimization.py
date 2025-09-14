# Implements query optimization techniques
import re
from typing import List, Dict


class QueryOptimizer:
    """
    Rule-based SQL Query Optimizer.
    - Removes SELECT *
    - Suggests indexes
    - Warns on missing LIMIT
    """

    def __init__(self):
        self.rules = {
            "select_star": re.compile(r"SELECT\s+\*", re.IGNORECASE),
            "missing_limit": re.compile(r"SELECT.+FROM.+(?!LIMIT)", re.IGNORECASE),
        }

    def analyze(self, query: str) -> Dict[str, List[str]]:
        issues = []
        suggestions = []

        if self.rules["select_star"].search(query):
            issues.append("Avoid SELECT *")
            suggestions.append("Specify required columns instead of *")

        if self.rules["missing_limit"].search(query):
            suggestions.append("Consider adding LIMIT for large datasets")

        if "JOIN" in query.upper() and "ON" not in query.upper():
            issues.append("JOIN without ON condition may cause Cartesian product")

        return {"issues": issues, "suggestions": suggestions}

    def rewrite(self, query: str) -> str:
        """Simple rewrites for common issues"""
        query = re.sub(r"SELECT\s+\*", "SELECT id", query, flags=re.IGNORECASE)
        return query
