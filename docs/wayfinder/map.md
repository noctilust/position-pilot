# Position Pilot Web Revamp

## Notes

- Canonical requirements: [Position Pilot Web Revamp PRD](../PRD.md)
- Delivery sequence: [Phased Implementation Plan](../IMPLEMENTATION_PLAN.md)
- Preserve read-only brokerage access and `.env` credential storage.
- Use the `implement`, `frontend-design`, and `code-review` skills during delivery.
- Local Markdown is the tracker because this repository defines no issue-tracker convention.

## Decisions so far

- [Confirmed product and domain requirements](tickets/000-confirmed-requirements.md) — The grilling session resolved product scope, providers, AI, cadence, privacy, reliability, and migration policy.
- [Web application foundation](tickets/010-web-foundation.md) — Phase 1 ships the secure local service, packaged workstation shell, web-default launcher, and regression gates.

## Fog

- Exact Tastytrade streaming event coverage and reconnect semantics need a focused implementation investigation in Phase 3.
- Provider licensing entitlements must be read from the user's purchased plans rather than inferred from public marketing pages.
- Codex CLI structured-output behavior and rate-limit signaling require a prototype before scheduler integration.
- Final TUI deletion depends on an explicit parity matrix and regression evidence.
