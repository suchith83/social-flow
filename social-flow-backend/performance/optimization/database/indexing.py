# Provides indexing utilities for faster lookups
from typing import List, Dict


class IndexAdvisor:
    """
    Advises on indexing strategy based on queries.
    """

    def __init__(self):
        self.index_suggestions: Dict[str, List[str]] = {}

    def analyze_query(self, table: str, query: str) -> List[str]:
        suggestions = []
        if "WHERE" in query.upper():
            suggestions.append("Create index on WHERE clause columns")
        if "ORDER BY" in query.upper():
            suggestions.append("Consider composite index on ORDER BY columns")
        if "JOIN" in query.upper():
            suggestions.append("Index join keys for better performance")

        self.index_suggestions[table] = suggestions
        return suggestions

    def get_recommendations(self) -> Dict[str, List[str]]:
        return self.index_suggestions
