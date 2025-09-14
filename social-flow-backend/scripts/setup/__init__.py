# scripts/setup/__init__.py
"""
Setup package for Social Flow (developer & host provisioning)

Provides utilities to:
 - Load environment-specific configuration
 - Bootstrap environment variables and .env files
 - Install OS and Python/Node dependencies
 - Configure Docker (login, daemon tweaks)
 - Configure kubectl / kubeconfig contexts for clusters
 - Create system users / groups for services
 - Manage TLS certificates (self-signed dev certs or request via ACME)
 - Bootstrap secrets to secret manager placeholders (safe, non-destructive)
 - Cleanup temporary bootstrap artifacts
 - A top-level runner to orchestrate repeatable idempotent setup runs

Authorship: Social Flow DevOps Team
"""
__version__ = "1.0.0"
__author__ = "Social Flow DevOps Team"
