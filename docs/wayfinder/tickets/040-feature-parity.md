---
status: closed
type: task
blocked_by:
  - Implement streaming and provider reconciliation
---

# Deliver full portfolio feature parity

## Question

How will every current portfolio, market, watchlist, roll, order, and analysis capability move into the web information architecture with calculation parity?

## Scope

Phase 4 exit requires verified web parity with the current CLI/TUI feature set:

- Positions, grouped strategies, Greeks, P/L, balances, and account summary
- Watchlist, market overview, quotes, and IV environment
- Roll history, chain credits, pattern analytics, and heatmap
- Read-only order activity and fill linkage
- Position detail with charts, events, thesis or trade plan, and audit history
- Stress scenarios and deterministic portfolio risk

Constraints:

- Reuse domain services and SQLite
- Tastytrade stays read-only and first provider
- Never expose brokerage identifiers, names, credentials, or env values
- Preserve TUI and untracked user work
- Defer Phase 5–7 (catalysts, Codex monitoring, TUI retirement)

## Resolution

Phase 4 is implemented on shared domain services over SQLite schema v4:

- **Risk** (`domain/risk.py`) — deterministic combined Greeks, concentration, mark-based remaining profit/loss bounds and breakevens for common structures, and standard price/IV/theta/expiration stress scenarios without AI.
- **Orders** (`domain/orders.py`) — read-only order history with hashed public order/fill IDs, fill linkage, and no brokerage account numbers in payloads.
- **Plans** (`domain/plans.py`) — strategic thesis, tactical trade plan, and immutable audit events persisted in SQLite.
- **Watchlist** (`domain/watchlist.py`) — SQLite-backed symbols with quote assembly from the market service.
- **Rolls** — continuously synchronized CLI/TUI history, chain credits, pattern analytics via existing `analyze_patterns`, and strike-by-DTE heatmap.
- **Markets** — overview, single quotes, IV environment classification, and chart bars (provider when available, latest mark fallback).
- **Web API** — `/api/v1` routes for portfolio risk, strategy detail, thesis/trade-plan/audit, markets, watchlist, orders, roll patterns/heatmap; bootstrap phase `portfolio-parity`.
- **Frontend** — Overview, Positions with orders, Roll analytics, Markets/watchlist, Settings, and strategy detail drawer with chart, legs, risk/stress, thesis/plan, events, and rolls.

## Exit evidence

- Targeted Ruff on Phase 4 modules: clean
- Full pytest: 70 passed
- Frontend typecheck: clean
- Frontend production build: packaged into `src/position_pilot/web/static/`
- Playwright browser smoke: positions, detail drawer, rolls, markets, and settings passed axe checks with no console errors
- Reviewed frontend resilience fixes: account changes clear stale roll/risk state, roll subrequests degrade independently, and detail refreshes are request-sequenced
- Live read-only smoke: 5 accounts, 26 strategies, 29 positions; risk, markets, and watchlist returned live data
- Two independent Grok reviews completed; confirmed roll-sync, Greek-scale, frontend race, form-parity, and accessibility findings were fixed and regression-tested
- `uv build`: wheel and sdist produced
- TUI preserved; untracked `AGENTS.md` left untouched
