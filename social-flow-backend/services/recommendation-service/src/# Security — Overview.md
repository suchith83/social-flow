# Security — Overview

This folder contains developer-facing security guidance and lightweight tools to help keep the repository and deployments secure.

Contents:
- policies.md — concise policies for access control, encryption, secrets, and incident handling.
- certs/README.md — guidance for TLS certificate generation and handling in local dev.
- secrets/README.md — guidance for secrets management and recommended workflows.
- scanners/scan_secrets.sh — a minimal repository secret scanner (grep-based) for CI smoke checks.

Quick actions:
- Run the lightweight secret scan locally:
  bash security/scanners/scan_secrets.sh

- Generate development TLS certs:
  scripts/generate_self_signed.sh

- Use a proper secrets manager (Vault, AWS Secrets Manager) in production — never commit secrets to the repo.
