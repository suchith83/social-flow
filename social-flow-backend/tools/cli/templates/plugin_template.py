"""
Plugin Template for CLI
-----------------------

This file demonstrates how to build CLI plugins for extending
functionality dynamically. Developers can place plugin files
in the `plugins` directory and register new commands.

Features:
- Dynamic discovery
- Modular CLI extension
- Example plugin command
"""

import click
import logging

logger = logging.getLogger("cli-plugin")


def register(cli):
    """
    Register new commands with the base CLI.
    Called dynamically at runtime when loading plugins.
    """

    @cli.command()
    @click.option("--name", prompt="Your name", help="Name for greeting.")
    def greet(name):
        """
        Example plugin command: prints a friendly greeting.
        """
        logger.info("👋 Hey %s, welcome to Social Flow CLI plugins!", name)
