"""
Lightweight Python parser to extract AST-level info used by rules.
We use the standard ast library for reliable parsing.
This parser intentionally focuses on structure extraction (imports, function defs, literals),
not on performing heavy static checks (that's the analyzers' job).
"""

import ast
from typing import Dict, Any, List
from ..utils import logger

class PythonParser:
    @staticmethod
    def parse_file(path: str) -> Dict[str, Any]:
        """
        Parse Python file into a small AST-derived structure that rules can consume.
        Returns dict with keys: "imports", "functions", "classes", "literals"
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=path)
        except Exception as e:
            logger.exception(f"Failed to parse {path}: {e}")
            return {"path": path, "error": str(e)}

        parser = _PythonASTVisitor()
        parser.visit(tree)
        return {
            "path": path,
            "imports": parser.imports,
            "functions": parser.functions,
            "classes": parser.classes,
            "string_literals": parser.string_literals,
            "has_suspicious_literals": parser.has_suspicious_literals
        }

class _PythonASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports: List[str] = []
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.string_literals: List[str] = []
        self.has_suspicious_literals = False

    def visit_Import(self, node: ast.Import):
        for n in node.names:
            self.imports.append(n.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for n in node.names:
            self.imports.append(f"{mod}.{n.name}" if mod else n.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions.append({
            "name": node.name,
            "args": [a.arg for a in node.args.args],
            "lineno": node.lineno
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append({"name": node.name, "lineno": node.lineno})
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self.string_literals.append(node.value)
            if len(node.value) > 20 and ("password" in node.value.lower() or "secret" in node.value.lower()):
                self.has_suspicious_literals = True
        self.generic_visit(node)

    # For Python <3.8 compatibility, also check Str nodes
    def visit_Str(self, node: ast.Str):
        self.string_literals.append(node.s)
        if len(node.s) > 20 and ("password" in node.s.lower() or "secret" in node.s.lower()):
            self.has_suspicious_literals = True
        self.generic_visit(node)
