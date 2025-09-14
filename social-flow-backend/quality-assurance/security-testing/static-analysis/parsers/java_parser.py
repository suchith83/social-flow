"""
Very basic Java parser using regex to find imports and string literals.
For real projects integrate with a Java parser (javaparser).
"""

import re
from typing import Dict, Any
from ..utils import logger

IMPORT_RE = re.compile(r'^\s*import\s+([a-zA-Z0-9\._\*]+);', re.MULTILINE)
STRING_RE = re.compile(r'"([^"\\]{3,})"')

class JavaParser:
    @staticmethod
    def parse_file(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as e:
            logger.exception(f"Failed to read {path}: {e}")
            return {"path": path, "error": str(e)}

        imports = IMPORT_RE.findall(src)
        strings = STRING_RE.findall(src)
        suspicious = any("password" in s.lower() or "secret" in s.lower() for s in strings)

        return {
            "path": path,
            "imports": imports,
            "string_literals": strings,
            "has_suspicious_literals": suspicious
        }
