"""
Forbidden Comments Checker

This is a RAW (token/line) checker that inspects source lines and flags
forbidden comment markers such as TODO, FIXME, HACK, and XXX.

Rationale:
  - Comments left as "TODO/FIXME/HACK" in production code are often
    forgotten and indicate incomplete work; raising awareness keeps the
    codebase cleaner.

Implementation notes:
  - Uses BaseChecker with IRawChecker or uses SimpleTokenChecker to access raw tokens.
  - We look at comment tokens only to avoid false positives inside strings.
"""

from typing import Iterable
from pylint.checkers import BaseChecker
from pylint.interfaces import IRawChecker
from pylint.lint import PyLinter

# Message id and symbol are intentionally specific and namespaced.
MSG_ID = "forbidden-comment"
MSG_SYMBOL = "forbidden-comment"
MSG = "Forbidden comment '%s' found. Resolve or track it in a proper issue tracker."

DEFAULT_FORBIDDEN = ("TODO", "FIXME", "HACK", "XXX")


class ForbiddenCommentsChecker(BaseChecker):
    """
    Raw checker that inspects comment tokens and reports forbidden patterns.
    """

    __implements__ = IRawChecker

    name = "forbidden-comments-checker"
    priority = -1  # run early/low priority
    msgs = {
        # (msg_id, msg_template, description)
        "C9901": (
            MSG,
            MSG_SYMBOL,
            "Detects forbidden comment markers such as TODO, FIXME, HACK, XXX",
        )
    }

    options = (
        (
            "forbidden-comments",
            {
                "default": ",".join(DEFAULT_FORBIDDEN),
                "type": "string",
                "metavar": "<comma separated>",
                "help": "Comma-separated list of forbidden comment markers to detect.",
            },
        ),
    )

    def __init__(self, linter: PyLinter = None):
        super().__init__(linter)
        # prepare set from option (linter will populate options automatically)
        # actual parsing is done in process_module because options are not guaranteed set at init
        self._forbidden = set(DEFAULT_FORBIDDEN)

    def process_module(self, node):
        """
        process_module is called with the module AST node. We fetch the raw
        file contents via linter to perform token-based scanning.
        """
        try:
            # linter._current_file is internal; instead we use node.file_stream or node.file
            filepath = getattr(node, "file", None)
            file_content = node.file_stream.read().decode("utf-8") if getattr(node, "file_stream", None) else None
        except Exception:
            file_content = None

        # If raw content is not available here, fallback to scanning using the linter's current module
        # Pylint also provides a method to iterate tokens via self.linter.current_file_tokens in some versions,
        # but to keep compatible we fallback to a safer path: iterate node.stream() if available.
        if file_content is None:
            # fallback: skip (best-effort)
            return

        # parse options
        raw_option = self.config.forbidden_comments or ""
        markers = [x.strip() for x in raw_option.split(",") if x.strip()]
        self._forbidden = set(m.upper() for m in markers) if markers else set(DEFAULT_FORBIDDEN)

        # Scan line-by-line and report
        for lineno, line in enumerate(file_content.splitlines(), start=1):
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                # skip non-comments
                continue
            # simple tokenization: look for markers in uppercase
            upper = stripped.upper()
            for marker in self._forbidden:
                if marker in upper:
                    # column: index within line (first occurrence)
                    col = stripped.upper().index(marker)
                    self.add_message(
                        "C9901",
                        line=lineno,
                        node=None,
                        args=(marker,),
                        col_offset=line.find(stripped) + col,
                    )
                    break
