"""
Complexity Checker

This checker computes cyclomatic complexity per function and reports when the
value exceeds a configurable threshold.

Implementation approach:
  - Use AST nodes to count decision points:
    - if / elif
    - for / while
    - except
    - boolean operators (and/or) inside conditions
    - ternary expressions
    - comprehensions with if
    - match/case (Python 3.10+)

This is not a perfect algorithm but a pragmatic and fast one suitable for CI gating.
"""

from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker
from astroid import nodes
import itertools
from pylint.lint import PyLinter

DEFAULT_THRESHOLD = 10

class ComplexityChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "complexity-checker"
    priority = -1
    msgs = {
        "R9903": (
            "Function '%s' has cyclomatic complexity %d (threshold %d).",
            "high-cyclomatic-complexity",
            "Function is too complex and should be refactored to reduce complexity.",
        )
    }

    options = (
        (
            "max-cyclomatic-complexity",
            {
                "default": DEFAULT_THRESHOLD,
                "type": "int",
                "metavar": "<int>",
                "help": "Maximum allowed cyclomatic complexity for functions.",
            },
        ),
    )

    def __init__(self, linter: PyLinter = None):
        super().__init__(linter)

    def visit_functiondef(self, node: nodes.FunctionDef):
        complexity = self._compute_complexity(node)
        threshold = int(self.config.max_cyclomatic_complexity or DEFAULT_THRESHOLD)
        if complexity > threshold:
            self.add_message("R9903", node=node, args=(node.name, complexity, threshold))

    def visit_asyncfunctiondef(self, node: nodes.AsyncFunctionDef):
        self.visit_functiondef(node)

    def _compute_complexity(self, node: nodes.FunctionDef) -> int:
        """
        Compute a heuristic cyclomatic complexity for the given function node.
        We count:
          - if / elif branches as +1 each
          - for / while loops as +1
          - boolean operators 'and'/'or' in conditions as +1 each
          - each except clause as +1
          - conditional expressions (ternary) as +1
          - comprehensions with if as +1
          - match/case (if available) as +1 per case
        Base complexity starts at 1.
        """
        complexity = 1

        for child in node.body:
            complexity += self._complexity_in_node(child)

        # Also analyze nested nodes recursively
        for descendant in node.nodes_of_class(
            (
                nodes.If,
                nodes.For,
                nodes.While,
                nodes.IfExp,
                nodes.TryExcept,
                nodes.TryFinally,
                nodes.With,
                nodes.BoolOp,
                nodes.GeneratorExp,
                nodes.ListComp,
                nodes.DictComp,
                nodes.SetComp,
                nodes.Match,  # Python 3.10+
            )
        ):
            complexity += self._complexity_in_node(descendant)

        return complexity

    def _complexity_in_node(self, node) -> int:
        """
        Return complexity contribution of a single node.
        """
        contrib = 0
        if isinstance(node, nodes.If):
            # each 'if' is +1; chained elifs appear as nested Ifs
            # Count branches: 1 for the if, and 1 for each elif block present
            contrib += 1
            # count boolean operators inside the test
            contrib += self._bool_ops_in(node.test)
            # if there are elifs, they appear as node.orelse containing another If in astroid
            # node.orelse processing is handled by recursion when scanning descendants
        elif isinstance(node, (nodes.For, nodes.While)):
            contrib += 1
        elif isinstance(node, nodes.IfExp):
            # ternary conditional
            contrib += 1
        elif isinstance(node, (nodes.GeneratorExp, nodes.ListComp, nodes.SetComp, nodes.DictComp)):
            # comprehension with conditional parts
            # count 'if' filters inside generators/comprehensions
            for gen in getattr(node, "generators", []) or []:
                if getattr(gen, "ifs", None):
                    contrib += len(gen.ifs)
        elif isinstance(node, nodes.TryExcept):
            # each except clause is a new control flow path
            contrib += len(node.handlers) if getattr(node, "handlers", None) else 0
        elif isinstance(node, nodes.BoolOp):
            # and/or increases complexity; count operations
            if getattr(node, "op", None):
                # node.values length - 1 is number of operations
                contrib += max(0, len(node.values) - 1)
        elif isinstance(node, nodes.Match):
            # match/case: each case adds a branch
            for case in getattr(node, "cases", []):
                contrib += 1
        else:
            # default: 0
            pass

        return contrib

    def _bool_ops_in(self, expr) -> int:
        """
        Count boolean 'and'/'or' operations inside an expression.
        """
        count = 0
        if isinstance(expr, nodes.BoolOp):
            count += max(0, len(expr.values) - 1)
        # recursively inspect children for nested BoolOps
        for child in getattr(expr, "get_children", lambda: [])():
            try:
                count += self._bool_ops_in(child)
            except Exception:
                # ignore nodes that aren't expressions we understand
                pass
        return count
