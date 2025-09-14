# QA: Pylint Custom Plugin & Config

This package provides a set of custom Pylint checkers and a recommended config for enterprise code-quality enforcement.

## Contents

- `pylint_plugins/` — custom plugin checkers:
  - `forbidden_comments.py` — flags TODO/FIXME/HACK in comments.
  - `naming_convention.py` — enforces naming conventions for variables/functions/classes.
  - `complexity_checker.py` — computes cyclomatic complexity per function.

- `.pylintrc`, `setup.cfg` and `pyproject.toml` — installation and configuration.

## Quickstart

1. Install locally (editable):
```bash
pip install -e .
