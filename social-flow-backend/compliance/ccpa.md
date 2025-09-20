# CCPA — Developer Guidance (Draft)

Scope
- Applies to services storing/processsing data of California residents.

Key obligations
- Provide consumers the right to opt-out of sale of personal data.
- Implement data access and deletion mechanisms.
- Disclose categories of personal data collected and purposes.

Developer guidance
- Maintain a registry of data categories collected per service (PII, identifiers, usage, analytics).
- Add feature flags and endpoints to support Do Not Sell & Opt-out flows.
- When processing third-party ad/sponsor integrations, ensure contractual controls and provide opt-out.
- Implement robust logging of data disclosure to third parties (who, what, when).

Requests handling
- Route CCPA requests to the privacy team; provide programmatic export/deletion endpoints for automation.

Notes
- Coordinate with Legal to finalize public notices and in-app disclosures.
