# Repository configuration layout

This directory contains environment and service configuration templates used across the monorepo.

Structure
- .env.example                     -> example common environment variables
- environments/
  - development.env                -> local development overrides
  - staging.env                    -> staging environment example
  - production.env                 -> production environment example (secrets omitted)
- databases/
  - postgres.yaml                  -> example DB connection settings (for IaC / templates)
  - redis.yaml                     -> example Redis connection settings
- security/
  - secrets.example.env            -> example secrets file (DO NOT COMMIT real secrets)

Usage
- Copy .env.example -> .env and tune values for local development.
- Use environments/*.env per deployment environment.
- For services, prefer reading env vars (DATABASE_URL, REDIS_URL, SF_SECRET_KEY).
- Keep real secrets in your secret manager (Vault, AWS Secrets Manager) and NOT the repo.

Developer tips
- Use `python-dotenv` in local dev to load .env files.
- CI should supply secrets via pipeline variables/secrets.
