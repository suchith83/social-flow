# Secrets Management — Guidance

Never commit secrets into source control. This document lists recommended approaches.

Local development
- Use a local `.env` file (copy from config/.env.example) and add it to `.gitignore`.
- Use python-dotenv or runtime environment variable injection for local testing.
- For local secret rotation, regenerate tokens and update the secrets manager or local .env.

CI/CD
- Store secrets in CI secret storage (GitHub Actions Secrets, GitLab CI variables).
- Use short-lived tokens and inject them at runtime.
- Avoid printing secrets in CI logs.

Production
- Use a centralized secret manager (HashiCorp Vault, AWS Secrets Manager).
- Grant access via IAM roles and service accounts, not long-lived keys.
- Audit access to secrets and rotate credentials regularly.

Verification
- Run a secret-scan in CI before merges (tools: git-secrets, truffleHog, detect-secrets).
- A lightweight scan is available at security/scanners/scan_secrets.sh but prefer dedicated SCA tools.
