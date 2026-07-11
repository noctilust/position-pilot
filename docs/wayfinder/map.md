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
- [Versioned domain snapshots](tickets/020-domain-snapshots.md) — Phase 2 establishes shared account, portfolio, strategy, market, and roll services over durable SQLite state.
- [Streaming and provider reconciliation](tickets/030-streaming-providers.md) — Phase 3 adds independently supervised Tastytrade streams, REST authority, field-level fallback routing, provider health, and redacted browser events.

## Fog

- Provider licensing entitlements must be read from the user's purchased plans rather than inferred from public marketing pages.
- Codex CLI structured-output behavior and rate-limit signaling require a prototype before scheduler integration.
- Final TUI deletion depends on an explicit parity matrix and regression evidence.
