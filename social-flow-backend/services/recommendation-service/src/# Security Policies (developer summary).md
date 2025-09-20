# Security Policies (developer summary)

This file summarizes core security policies engineers must follow. It is not legal advice — consult Security/Legal for formal policies.

1) Access Control & Identity
- Principle: least privilege.
- Use role-based access: user, creator, moderator, admin, infra.
- Store role/permission mapping centrally (config or IAM).
- Require MFA for admin accounts and for production console access.
- Rotate service credentials on role changes or compromises.

2) Secrets & Credentials
- NEVER commit secrets to Git. Add any local secrets files to .gitignore.
- Use a secrets manager for production (AWS Secrets Manager, Vault, GCP Secret Manager).
- Limit secrets scope and use short-lived credentials where possible.
- Use environment variables or mounted secrets for runtime (do not bake into images).

3) Encryption
- Transport: enforce TLS for all internal/external traffic.
- At-rest: enable encryption for databases, object storage and backups.
- Key management: use KMS (AWS KMS, GCP KMS). Rotate keys per policy.

4) Certificates
- For dev: use self-signed certs and store under .certs/ (ignore in VCS).
- For production: use managed certificates (ACM) and automate renewal.

5) Logging & Audit
- Emit structured audit logs for privileged actions (role changes, data exports, takedowns).
- Protect log integrity and restrict access to audit logs.

6) Dependency & Supply Chain
- Keep dependencies up to date. Use automated SCA (Dependabot, GitHub Advanced Security).
- Verify and pin container base images; scan images for vulnerabilities.

7) CI/CD
- Do not expose secrets in CI logs.
- Use least-privilege build agents and ephemeral credentials.
- Require signed commits or trusted CI jobs for production deploys.

8) Incident Response
- Immediately rotate compromised keys or tokens.
- Follow the incident_response playbook in compliance/incident_response.md.
- Preserve evidence and notify Security & Legal per policy.

9) Compliance
- Follow data retention and privacy policies (see compliance/).
- For personal data, pseudonymize in analytics pipelines.

If a rule conflicts with developer productivity, raise it in Security Slack/PR — do not circumvent policies by committing secrets or hardcoded credentials.
