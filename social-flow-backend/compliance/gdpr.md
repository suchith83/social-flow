# GDPR — Developer Guidance (Draft)

Purpose
- Provide practical guidance for engineering teams handling personal data of EU data subjects.

Principles (apply to all services)
- Lawfulness, fairness, transparency
- Purpose limitation
- Data minimization
- Accuracy
- Storage limitation
- Integrity and confidentiality
- Accountability

Developer responsibilities
- Only collect fields that are required for the feature; document justification.
- Use environment-aware configs to enable stricter defaults for EU deployments.
- Store personal identifiers (email, phone, IP) in encrypted form at rest when feasible.
- Use pseudonymization for analytics and ML datasets (remove direct identifiers).
- Implement deletion flows: accept deletion requests and cascade to all services (DBs, caches, backups).
- Log data access events to audit logs (see audit_logging.md).

Data subject rights (implement)
- Right to access: provide endpoint or admin tooling to export a user's data.
- Right to rectification: allow updates to personal data.
- Right to erasure: delete or anonymize data on request, including backups where possible.
- Right to portability: export data in JSON/CSV.

Breach notifications
- Notify security and legal teams immediately per incident_response.md if personal data breach is suspected.

Notes
- This document is a practical developer guide. Legal should review before publishing external policy.
