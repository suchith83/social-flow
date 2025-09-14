# scripts/maintenance/__init__.py
"""
Maintenance package for Social Flow

Provides:
- Backup management (S3/on-disk)
- Database maintenance (vacuum, optimize, migrate)
- Log rotation & archival
- Cleanup of temporary files, caches, old deployments
- Scheduler wrapper to run tasks (cron/systemd-friendly)
- Notifier for Slack/Email
- Locking to prevent concurrent runs
"""

__version__ = "1.0.0"
__author__ = "Social Flow DevOps Team"
