# common/libraries/python/database/query_builder.py
"""
Lightweight SQL query builder to prevent SQL injection.
"""

from sqlalchemy import text

class QueryBuilder:
    @staticmethod
    def select(table: str, columns: list[str] = None, where: str = None):
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table}"
        if where:
            query += f" WHERE {where}"
        return text(query)

    @staticmethod
    def insert(table: str, values: dict):
        cols = ", ".join(values.keys())
        params = ", ".join([f":{k}" for k in values.keys()])
        return text(f"INSERT INTO {table} ({cols}) VALUES ({params})").bindparams(**values)

    @staticmethod
    def update(table: str, values: dict, where: str):
        set_clause = ", ".join([f"{k}=:{k}" for k in values.keys()])
        return text(f"UPDATE {table} SET {set_clause} WHERE {where}").bindparams(**values)

    @staticmethod
    def delete(table: str, where: str):
        return text(f"DELETE FROM {table} WHERE {where}")
