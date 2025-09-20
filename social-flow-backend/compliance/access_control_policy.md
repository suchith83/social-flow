# Access Control & RBAC — Guidance

Principles
- Least privilege by default
- Role-based access control with scoped permissions
- Separation of duties for sensitive operations
- Just-in-time elevated access for emergency tasks

Roles (examples)
- user — standard product user
- creator — content uploader with monetization permissions
- moderator — content moderation privileges (takedown, warnings)
- admin — system admin, limited to operational tasks; require MFA
- legal/security — access to sensitive logs and compliance workflows

Developer guidance
- Implement RBAC at API gateway / service middleware layer.
- Use short-lived tokens for elevated actions and require MFA for admin workflows.
- Audit all role changes and privileged actions.

Secrets & credential handling
- Store secrets in a secrets manager (AWS Secrets Manager, HashiCorp Vault).
- Avoid embedding secrets in code or config repos.

Notes
- Map concrete permissions to roles in a central permissions matrix and store in a configuration service or database for runtime enforcement.
