---
status: closed
type: prototype
blocked_by:
  - Implement streaming and provider reconciliation
---

# Build evidence-backed catalyst intelligence

## Question

How should source ingestion, deduplication, attribution, abstention, option-market drivers, charts, retention, and feedback combine into a trustworthy catalyst experience?

## Scope

Phase 5 exit requires every unique held underlying to receive an evidence-backed catalyst result or an explicit `No confirmed catalyst found` abstention:

- Normalized catalyst/source models, taxonomy, confidence, attribution, provenance/freshness/coverage
- Held-underlying scan with previous-close-to-now window (extended hours), up to 72h for scheduled events, stock ≥2% / broad ETF ≥1% thresholds, abnormal/high-impact promotion
- Source priority: Massive baseline news + configuration-gated Benzinga; independent provider failures; dedupe without collapsing distinct events
- Company/peer/macro/options-market attribution; option mechanisms clearly labeled as mechanisms, never news
- Deterministic evidence ranking and abstention; social only as non-evidence side note
- SQLite schema v5 with one-year catalyst/source retention, 90-day metadata/excerpt retention, license-dependent full text + removal handling
- Feedback Relevant / Not related / Missing catalyst with immutable history and local ranking adjustment only
- Web API + frontend UX on overview, positions, and strategy detail with independent freshness/offline/incomplete-coverage state
- Intraday extended-hours charts with volume, official prior close, and event markers
- Configuration for news cadence and Benzinga status; never expose credentials

## Resolution

Phase 5 is implemented on shared domain services over SQLite schema v5, with review corrections applied:

- **Domain** (`domain/catalysts.py`) — corrected prior-close session math; strong abstention; provider state matrix (not_configured / healthy empty / degraded / offline); offline STALE cache; stable source IDs; feedback ranking with symbol resolution; sole-causal-proof guard for incomplete coverage; option mechanism coverage ledger.
- **Production composition** (`domain/factory.py`) — never treats live quote `close` as prior close; derives prior close and session high/low from extended-hours bars; runtime Benzinga toggle via provider factory; aggregates Massive option metrics only from real held contract symbols, with explicit unavailable coverage.
- **Persistence** — schema v5; idempotent sources; snapshot pruning; removal deletes licensed article/source material and unsupported derived presentation without mutating feedback.
- **Charts** — prior-close-to-now window, truthful extended-hours flag, sampled high-density volume rendering, and an accessible recent-volume table.
- **Web/API/Frontend** — retention on app lifespan; feedback validation; account-switch clears catalysts; every held symbol surfaces explicit abstention (never bare dash); accessible volume series.

## Exit evidence (post-review)

- Focused catalyst unit/API tests including prior-close phases, bar-derived moves, production composition seam, offline cache, Benzinga toggle, strong abstention, stable sources, feedback ranking, sole-causal guard, mechanism coverage, retention/removal
- Full pytest: **101 passed**
- Ruff clean on Phase 5 modules
- Frontend typecheck: clean
- Frontend production build: packaged into `src/position_pilot/web/static/`
- Playwright smoke/accessibility test: **1 passed**
- Python sdist and wheel build: clean
- TUI preserved; no Phase 6/7 work; AGENTS.md untouched

## Honest limitations

- Exchange-holiday calendar is still simplified (weekend-aware prior close only; no holiday calendar).
- Unusual volume/OI/skew/gamma depend on provider fields; when absent they are explicitly marked **unavailable** rather than invented.
- Public-web supplementation is intentionally not implemented.
- News cadence is stored/exposed; background polling remains Phase 6.
- Live Benzinga/Massive network behavior is mocked in tests; real entitlements remain operator-configured via `.env`.
- Massive option metrics cover held contracts in the latest portfolio snapshot; whole-chain market aggregates remain provider-dependent.
