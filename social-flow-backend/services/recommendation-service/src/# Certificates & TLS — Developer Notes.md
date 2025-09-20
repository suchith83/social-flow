# Certificates & TLS — Developer Notes

Purpose
- Document how to manage TLS certs for local development and production.

Local development
- Use the provided script to generate a self-signed cert for localhost:
  scripts/generate_self_signed.sh
- Store development certs under `.certs/` and add `.certs/` to .gitignore.
- Do not use dev certs in production.

Production
- Use managed TLS (AWS ACM, Google-managed certificates) or obtain certificates from a CA.
- Automate issuance and renewal (ACME / cert-manager for Kubernetes).
- Store private keys in a secrets manager and restrict access.

Best practices
- Enforce TLS for all internal and external endpoints.
- Use HSTS, secure cookies, and TLSv1.2+.
- Monitor certificate expirations and automate alerts for renewals.
