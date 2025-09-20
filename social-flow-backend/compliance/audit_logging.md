# Audit Logging — Guidance

Purpose
- Ensure privileged actions and sensitive data accesses are logged for audit and incident response.

What to log
- Authentication events: login success/failure, password change, token issuance, revocation.
- Data access: exports, deletions, bulk reads of personal data.
- Administrative actions: permission changes, role assignments, takedowns.
- Security events: suspicious activity, failed integrity checks, anomalous API usage.

Log contents (minimum)
- Actor (user_id or service id)
- Action type
- Resource identifier(s)
- Timestamp (ISO 8601)
- Outcome (success/failure) and reason for failure
- Correlation id / request id

Retention & access
- Retain audit logs per data_retention_policy.md.
- Restrict access to logs via RBAC; use separate logging pipeline for sensitive events.
- Protect logs with integrity controls (WORM, append-only storage, checksums).

Developer responsibilities
- Use structured logs (JSON) with consistent field names.
- Do not log raw secrets, PII or full payment info.
- Emit a request-id for every inbound request and include it in logs.

Notes
- Consider shipping audit logs to a secure, tamper-resistant store (e.g., CloudWatch Logs with encryption and restricted IAM).
