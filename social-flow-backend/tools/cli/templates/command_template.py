#!/usr/bin/env python3
"""
Command Template for CLI
------------------------

This file serves as a boilerplate for creating new CLI commands
in the Social Flow ecosystem. It uses `click` for rich command-line
interfaces, automatic help generation, colorized output, and subcommands.

Developers can copy this template, rename the command, and
extend functionality as required.

Features:
- Structured argument parsing
- Rich logging
- Config integration
- Error handling
- Extensible via plugins
"""

import click
import logging
import sys
from pathlib import Path
import yaml

# Configure logging with colored output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cli-template")

CONFIG_FILE = Path.home() / ".socialflow" / "config.yaml"


def load_config():
    """
    Load configuration from YAML file.
    Falls back to defaults if not found.
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error("Invalid YAML in config: %s", e)
            sys.exit(1)
    return {"environment": "development", "debug": False}


@click.group()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.pass_context
def cli(ctx, debug):
    """
    Root CLI entrypoint for new commands.
    """
    ctx.ensure_object(dict)
    ctx.obj["CONFIG"] = load_config()
    ctx.obj["DEBUG"] = debug or ctx.obj["CONFIG"].get("debug", False)
    if ctx.obj["DEBUG"]:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")


@cli.command()
@click.argument("name")
@click.option("--uppercase", is_flag=True, help="Print in uppercase.")
@click.pass_context
def hello(ctx, name, uppercase):
    """
    Example command.
    Greets a user by name with optional uppercase transformation.
    """
    greeting = f"Hello, {name}!"
    if uppercase:
        greeting = greeting.upper()
    logger.info(greeting)


@cli.command()
@click.option("--times", default=1, show_default=True, help="How many times to repeat.")
@click.pass_context
def repeat(ctx, times):
    """
    Example command that repeats a message.
    """
    for i in range(times):
        logger.info("This is repetition #%s", i + 1)


if __name__ == "__main__":
    cli(obj={})
