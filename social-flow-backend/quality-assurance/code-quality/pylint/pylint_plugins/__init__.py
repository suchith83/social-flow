


## 5) `pylint_plugins/__init__.py`

"""
pylint_plugins package entrypoint.

This module registers all custom checkers with Pylint. Pylint discovers plugins
by importing this package and calling `register` function (or using entry_points).

We expose a `register` function to be compatible with both plugin-loading
mechanisms (entry points or load-plugins).
"""

from pylint.lint import PyLinter

# import checkers so we can register them
from .forbidden_comments import ForbiddenCommentsChecker
from .naming_convention import NamingConventionChecker
from .complexity_checker import ComplexityChecker


def register(linter: PyLinter) -> None:
    """
    Called by pylint to register our custom checkers.

    :param linter: PyLinter instance provided by Pylint
    """
    linter.register_checker(ForbiddenCommentsChecker(linter))
    linter.register_checker(NamingConventionChecker(linter))
    linter.register_checker(ComplexityChecker(linter))
