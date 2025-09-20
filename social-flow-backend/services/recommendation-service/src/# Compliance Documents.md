# Compliance Documents

This folder contains concise, developer-facing compliance policies and templates
used by the Social Flow Backend project. These are intended as living documents:
they should be reviewed by legal, security and privacy teams and integrated into
onboarding, audits and CI/CD gating.

Files:
- gdpr.md — Data protection principles and developer responsibilities for EU data.
- ccpa.md — Consumer privacy obligations for California residents.
- dmca.md — Takedown & counter-notice process for copyright claims.
- privacy_policy.md — High-level privacy commitments for public facing docs.
- data_retention_policy.md — Retention and deletion rules for user and analytics data.
- audit_logging.md — Logging, retention and access controls for audit logs.
- incident_response.md — Incident triage and notification steps.
- access_control_policy.md — RBAC and least-privilege guidance.
- compliance_checklist.yml — Machine-friendly checklist to integrate into CI.

How to use:
- Developers: follow "Developer responsibilities" sections.
- DevOps: reference checklist in CI pipelines to enforce config and secrets handling.
- Legal/Security: review and approve; convert to formal policies if required.
