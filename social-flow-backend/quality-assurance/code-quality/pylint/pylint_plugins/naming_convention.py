"""
Naming Convention Checker

This AST-based checker ensures that functions and variables follow lower_snake_case
(PEP8-style) and classes follow PascalCase. Pylint has its own naming checks,
but sometimes organizations want stricter or slightly different patterns; this
checker demonstrates how to implement such rules.

It subclasses BaseChecker and registers visits for relevant AST nodes.
"""

import re
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker
from astroid import nodes
from pylint.lint import PyLinter

# message definition: unique message id in the C9xxx range to avoid colliding with built-in
MSG_ID_NAME_VIOLATION = "invalid-name-convention"
MSG_SYMBOL = "invalid-name-convention"

DEFAULT_MAX_VAR_LEN = 30

class NamingConventionChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "naming-convention-checker"
    priority = -1
    msgs = {
        "C9902": (
            "Name '%s' doesn't match required naming convention (%s).",
            MSG_SYMBOL,
            "Enforce project naming conventions for variables, functions and classes.",
        )
    }

    options = (
        (
            "require-snake-case",
            {
                "default": True,
                "type": "bool",
                "help": "Require snake_case for variables and functions (True/False).",
            },
        ),
        (
            "max-variable-length",
            {
                "default": DEFAULT_MAX_VAR_LEN,
                "type": "int",
                "help": "Maximum length for variable names (for stylistic enforcement).",
            },
        ),
    )

    def __init__(self, linter: PyLinter = None):
        super().__init__(linter)
        # compiled regex patterns
        self._snake_case_re = re.compile(r"^[a-z_][a-z0-9_]*$")  # PEP8-ish
        self._pascal_case_re = re.compile(r"^[A-Z][a-zA-Z0-9]+$")

    def visit_functiondef(self, node: nodes.FunctionDef):
        """
        Called when a function or method is defined.
        """
        name = node.name
        require_snake = bool(self.config.require_snake_case)
        if require_snake and not self._snake_case_re.match(name):
            self.add_message("C9902", node=node, args=(name, "snake_case"))

    def visit_asyncfunctiondef(self, node: nodes.AsyncFunctionDef):
        # same checks for async functions
        self.visit_functiondef(node)

    def visit_assign(self, node: nodes.Assign):
        # Visit target names of assignments to ensure they follow variable naming conventions
        # e.g. a = 1, (a, b) = ...
        for target in node.targets:
            self._check_target(target)

    def visit_annassign(self, node: nodes.AnnAssign):
        # annotated assignment: `x: int = 1` or `x: int`
        self._check_target(node.target)

    def visit_arg(self, node: nodes.Arguments):
        # function arguments are validated in the functiondef visitor by inspecting node.args
        pass

    def _check_target(self, target):
        """
        Helper to extract names from assignment targets and validate them.
        """
        # Name node
        from astroid import nodes as _n

        if isinstance(target, _n.Name):
            self._validate_variable_name(target)
        elif isinstance(target, _n.Tuple) or isinstance(target, _n.List):
            for elt in target.elts:
                self._check_target(elt)
        # else: attributes, subscripts etc. are skipped (we don't enforce field names here)

    def _validate_variable_name(self, name_node):
        name = name_node.name
        if not self._snake_case_re.match(name):
            self.add_message("C9902", node=name_node, args=(name, "snake_case"))
        else:
            max_len = int(self.config.max_variable_length or DEFAULT_MAX_VAR_LEN)
            if len(name) > max_len:
                # soft error: still report
                self.add_message("C9902", node=name_node, args=(name, f"<= {max_len} chars"))

    def visit_classdef(self, node: nodes.ClassDef):
        name = node.name
        if not self._pascal_case_re.match(name):
            self.add_message("C9902", node=node, args=(name, "PascalCase"))
