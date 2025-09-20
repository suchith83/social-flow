# Data Retention & Deletion Policy (Draft)

Purpose
- Define retention windows and deletion semantics for different data categories.

Retention windows (examples)
- Authentication data (users, credentials): retain until account deletion + 90 days for audit.
- Content (posts, videos): retain until user deletes; backups may exist for up to 180 days.
- Analytics & aggregated metrics: retain aggregated form indefinitely; raw event logs: 365 days.
- Logs & audit trails: retain 365 days (or per legal/regulatory requirements).
- Payment records (non-sensitive): retain 7 years (per accounting requirements).

Deletion semantics
- Soft-delete first (mark deleted), purge background job to remove PII fields after retention window.
- Anonymization preferred when full deletion would break analytics; remove direct identifiers and keep aggregated metrics.
- Ensure deletion cascades: DB records, caches, search indexes, object storage, CDNs, backups (where possible).

Developer guidance
- Implement deletion endpoint that enqueues a deletion job and returns a job ID.
- Record deletion events in audit logs with job outcome and timestamps.
- Document retention policies in service README and align with compliance_checklist.yml.

Notes
- Legal/regulatory requirements may override these examples. Consult legal for final retention schedules.
