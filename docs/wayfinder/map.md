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
- [Portfolio feature parity](tickets/040-feature-parity.md) — Phase 4 delivers web parity for positions, strategies, Greeks, P/L, balances, watchlist, markets, rolls, orders, position detail, thesis/trade plan, audit, and deterministic stress risk.
- [Catalyst intelligence](tickets/050-catalysts.md) — Phase 5 delivers evidence-backed held-symbol catalysts with abstention, attribution, option-market mechanisms, feedback, retention, charts, and configuration-gated Benzinga coverage.
- [Codex recommendations and monitoring](tickets/060-codex-monitoring.md) — Phase 6 replaces production Anthropic recommendations with a local Codex CLI provider, horizon-aware fingerprints, consent-gated monitoring, alerts, and immutable history/decisions.
- [Hardening and TUI retirement](tickets/070-hardening-retirement.md) — Phase 7 retires the Textual dashboard and Anthropic path, ships operations (export/backup/retention/diagnostics/update readiness), accessibility and scale validation, and packaging/migration polish.

## Fog

- Provider licensing entitlements must be read from the user's purchased plans rather than inferred from public marketing pages.
- Live Codex structured-output and rate-limit signaling are mocked in tests; real CLI versions may differ slightly.
- Monitoring calendar is algorithmic for standard NYSE holidays/early closes; ad-hoc emergency closures are not modeled.
- Wake/recovery uses scheduler time-gap and portfolio loader transitions (no OS sleep/network APIs).
- API-key recommendation fallback is intentionally non-functional and not user-selectable.
- Exchange-holiday calendar for prior-close catalyst windows is still approximate (weekend-aware only).
- Phase 7 PDF export is a minimal valid text PDF; full print-layout engines are out of scope.
