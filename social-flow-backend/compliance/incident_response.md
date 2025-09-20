# Incident Response — Playbook (Developer-focused)

Objective
- Provide quick triage steps for suspected security or privacy incidents.

Initial triage
1. Contain: if active compromise, isolate affected services (remove keys, stop ingestion).
2. Preserve evidence: snapshot logs, dump relevant DBs, record timestamps and processes.
3. Notify: inform Security and Legal teams immediately (pager/email/Slack channel).
4. Assess impact: identify affected data types and number of users.

Communication
- Follow internal communication policy: a single source (incident channel) and a designated incident commander.
- For breaches involving personal data, follow GDPR/CCPA notification timelines after Legal assessment.

Remediation
- Patch root cause, rotate keys/credentials, deploy fixes.
- Re-run integrity checks and increase monitoring for recurrence.

Post-incident
- Produce a post-incident report covering timeline, root cause, impact, remediation steps, and follow-up actions.
- Update policies & tests to prevent recurrence.

Developer tasks
- Add instrumentation to reproduce and verify fixes.
- Ensure incident playbook steps are automated where possible (scripts to rotate keys, block IPs).

Notes
- This playbook is a high-level starter; Security and Legal must own and maintain the official incident response policy.
