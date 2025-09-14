"""
Tests for custom Pylint plugins.

These tests run Pylint programmatically on a small snippet and assert
that expected messages are reported.

Note: These are integration-style tests. For more robust isolated testing,
you can use astroid to build AST nodes and run checker methods directly.
"""

import json
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO
import pytest

SAMPLE_CODE = r'''
# TODO: this should be removed
class badClassname:
    def BadMethodName(self, ARG):
        # FIXME: ugly hack
        for i in range(3):
            if i > 0 and (i % 2 == 0 or i == 1):
                print("debug")
        try:
            1/0
        except Exception as e:
            pass

def good_function_name(x, y):
    return x + y
'''

def run_pylint_on_code(code: str, extra_args=None):
    """
    Run pylint on given code string and return parsed messages (list of dicts).
    """
    buf = StringIO()
    reporter = TextReporter(buf)
    # write code to a temporary file
    from tempfile import NamedTemporaryFile
    tmp = NamedTemporaryFile("w+", suffix=".py", delete=False)
    tmp.write(code)
    tmp.flush()
    tmp.close()
    args = [tmp.name, "--load-plugins=pylint_plugins", "--output-format=json"]
    if extra_args:
        args.extend(extra_args)
    result = Run(args, reporter=reporter, do_exit=False)
    # reporter writes nothing in this flow because we used JSON output; read generated file
    # Pylint writes JSON to stdout; Run captures it internally for us accessible via result.linter
    # but simplest: call Run with --output-format=json and capture stdout via reporter (text reporter doesn't)
    # Instead run a subprocess-like approach using Run and then get raw messages from result.linter
    # Workaround: call Run and then obtain messages from result.linter.reporter. However reporter may not store them.
    # Simpler: read result.linter.stats and result.linter.reporter
    # To keep test robust we call Pylint with printed JSON to stdout by calling Run and checking exit status file.
    # But here we will call Pylint again using Run and parse its stdout by invoking its command-line interface via system call.
    import subprocess, sys
    completed = subprocess.run([sys.executable, "-m", "pylint", tmp.name, "--load-plugins=pylint_plugins", "--output-format=json"], capture_output=True, text=True)
    if completed.returncode not in (0, 32, 4, 16, 2):  # pylint exit codes vary; don't fail on non-zero here
        pass
    try:
        messages = json.loads(completed.stdout or "[]")
    except Exception:
        messages = []
    return messages

def test_forbidden_comments_detected():
    messages = run_pylint_on_code(SAMPLE_CODE)
    # find any message with our forbidden-comment symbol
    found = any(m.get("symbol") == "forbidden-comment" for m in messages)
    assert found, "Expected forbidden-comment message not found"

def test_naming_convention_detected():
    messages = run_pylint_on_code(SAMPLE_CODE)
    found = any(m.get("symbol") == "invalid-name-convention" for m in messages)
    assert found, "Expected invalid-name-convention message not found"

def test_complexity_detected():
    # set threshold low so sample triggers it
    messages = run_pylint_on_code(SAMPLE_CODE, extra_args=["--max-cyclomatic-complexity=2"])
    found = any(m.get("symbol") == "high-cyclomatic-complexity" for m in messages)
    assert found, "Expected high-cyclomatic-complexity message not found"
