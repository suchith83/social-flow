"""
A light JavaScript parser.
For robust JS/TS analysis, integrate with a JS AST tool (esprima/meriyah) or call eslint with AST extraction.
Here we keep a small regex-based extraction for quick rules (imports, string literals).
"""

import re
from typing import Dict, Any, List
from ..utils import logger

IMPORT_RE = re.compile(r'import\s+(?:.+\s+from\s+)?[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\)')

STRING_RE = re.compile(r'["\']([^"\']{3,})["\']')

class JavaScriptParser:
    @staticmethod
    def parse_file(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as e:
            logger.exception(f"Failed to read {path}: {e}")
            return {"path": path, "error": str(e)}

        imports = []
        for m in IMPORT_RE.finditer(src):
            imp = m.group(1) or m.group(2)
            if imp:
                imports.append(imp)

        string_literals = [m.group(1) for m in STRING_RE.finditer(src)]
        suspicious = any("password" in s.lower() or "secret" in s.lower() for s in string_literals)

        return {
            "path": path,
            "imports": imports,
            "string_literals": string_literals,
            "has_suspicious_literals": suspicious
        }
