# Tools — Developer utilities

This package contains small utilities and a CLI to help developers and CI run common tasks:
- Run tests and linters
- Aggregate runtime requirements across services
- Run docker-compose dev stacks via wrapper
- Helpers for automation scripts to import

Usage examples:
- Run the CLI (help): python -m tools.cli.main --help
- Run tests for a path: python -m tools.cli.main test --path services/recommendation-service
- Aggregate requirements: python -m tools.cli.main agg-reqs --out /tmp/reqs.txt

The tools are intentionally lightweight and defensive; they will print actionable instructions when external tools are not installed.
