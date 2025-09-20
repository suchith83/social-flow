"""
Unit Test Template for CLI Commands
-----------------------------------

This file contains a baseline test structure for new CLI commands.
It uses `pytest` and `click.testing.CliRunner` for easy test execution.
"""

import pytest
from click.testing import CliRunner
from command_template import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_hello_command(runner):
    result = runner.invoke(cli, ["hello", "World"])
    assert result.exit_code == 0
    assert "Hello, World!" in result.output


def test_hello_uppercase(runner):
    result = runner.invoke(cli, ["hello", "ChatGPT", "--uppercase"])
    assert result.exit_code == 0
    assert "HELLO, CHATGPT!" in result.output


def test_repeat_command(runner):
    result = runner.invoke(cli, ["repeat", "--times", "2"])
    assert result.exit_code == 0
    assert "This is repetition #1" in result.output
    assert "This is repetition #2" in result.output
