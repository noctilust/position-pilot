# Position Pilot Phased Implementation Plan

This plan delivers the confirmed PRD incrementally. Each phase must preserve a runnable application and add regression coverage before replacing existing behavior.

## Phase 1 — Web foundation and regression seams

**Goal:** Establish a production-shaped local web application without removing current functionality.

- Add backend test tooling and baseline service tests.
- Create a FastAPI application factory and versioned `/api/v1` routes.
- Add secure loopback defaults, health metadata, and no-store headers for sensitive endpoints.
- Create a React/TypeScript frontend with a coherent workstation design system.
- Package compiled frontend assets in the Python distribution.
- Make `pilot dashboard` launch the local web app while preserving `--tui` as a temporary fallback.
- Add local browser smoke coverage and accessibility checks for the shell.

**Exit:** The web dashboard launches reliably, exposes no secrets, builds reproducibly, and the legacy TUI remains reachable.

## Phase 2 — Domain extraction, SQLite, and snapshot model

**Goal:** Move business logic out of presentation code and establish durable, versioned state.

- Extract account, portfolio, strategy, roll, and market application services.
- Define field-level provenance and freshness models.
- Add SQLite schema, migrations, backup policy, and legacy-cache import.
- Implement atomic portfolio snapshots and offline-cache reads.
- Add account switcher and consolidated view without cross-account strategy grouping.
- Move settings into SQLite while keeping credentials in `.env`.

**Exit:** Both CLI and web UI consume the same tested services and versioned portfolio snapshots.

## Phase 3 — Streaming, reconciliation, and provider framework

**Goal:** Deliver responsive live state without silent drift.

- Implement Tastytrade DXLink market streaming.
- Implement available account-data streaming.
- Add five-minute REST reconciliation and gap detection.
- Define `MarketDataProvider`, `OptionsDataProvider`, `NewsProvider`, and provider-health contracts.
- Add Tastytrade-first field routing and explicit fallback behavior.
- Add Massive Stocks/News and Massive Options adapters behind configuration.

**Exit:** Live updates, reconnects, stale states, and provider failures are observable and isolated.

## Phase 4 — Full portfolio feature parity

**Goal:** Move every current feature into the web experience.

- Positions, grouped strategies, Greeks, P/L, balances, and account summary.
- Watchlist, market overview, quotes, and IV environment.
- Roll history, chain credits, pattern analytics, and heatmap.
- Read-only order activity and fill linkage.
- Position detail with charts, events, thesis or trade plan, and audit history.
- Stress scenarios and deterministic portfolio risk.

**Exit:** The web application has verified parity with the current CLI/TUI feature set.

## Phase 5 — Catalyst intelligence

**Goal:** Explain meaningful movement without manufacturing causality.

- Implement held-underlying scan, time windows, thresholds, and event taxonomy.
- Add primary-source and licensed-news ingestion with deduplication.
- Add company, peer, macro, and options-market attribution.
- Add full evidence provenance, confidence, abstention, feedback, and retention rules.
- Add intraday charts with extended hours, volume, prior close, and event markers.
- Add configurable news cadence and premium Benzinga integration.

**Exit:** Every held symbol has an evidence-backed result or an explicit abstention.

## Phase 6 — Codex recommendations and monitoring

**Goal:** Replace Anthropic and deliver horizon-aware, event-responsive recommendations.

- Implement a structured Codex CLI provider without token access.
- Add strategic and tactical horizon models, theses, and trade plans.
- Add prompt minimization, schema validation, status, retry, and explicit fallback selection.
- Add input fingerprints, 30-minute reevaluation, 60-second tactical triggers, and material-change detection.
- Add background scheduler, market calendar, wake/network recovery, and privacy-safe notifications.
- Add immutable recommendation history and trader decision tracking.

**Exit:** Recommendations are timely, auditable, subscription-aware, and never block core data.

## Phase 7 — Hardening, migration, and TUI retirement

**Goal:** Make the web application the only dashboard with safe migration and operational polish.

- Complete WCAG 2.2 AA, keyboard, responsive, visual-regression, and reduced-motion audits.
- Add exports, diagnostic bundles, retention controls, backup restore, and update workflow.
- Validate scale targets and performance budgets.
- Run live read-only smoke tests and calculation regression suites.
- Remove the TUI after web parity and migration checks pass.
- Correct Anthropic/Codex documentation and remove deprecated dependencies.

**Exit:** The web dashboard is the verified default and the legacy TUI is removed.

## Delivery rules

- Never request or use Tastytrade write scope.
- Never expose `.env` values to frontend payloads, logs, exports, or diagnostics.
- Do not present provider or AI guesses as confirmed facts.
- Preserve a working application at every phase boundary.
- Add tests at each extracted seam before changing behavior.
- Review and commit each phase independently.
