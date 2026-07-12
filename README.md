# Position Pilot

Position Pilot is a local, **read-only** portfolio workstation for Tastytrade options traders. The primary interface is a loopback web dashboard. The Python CLI remains available for quick commands and automation.

All trading recommendations are decision support only. The application never creates, stages, submits, replaces, or cancels broker orders.

## Installation

```bash
# Python + backend deps
uv sync --group dev
```

Normal installations use the **prebuilt React frontend** shipped inside the Python package. Contributors rebuild it with a pinned Node and pnpm toolchain (see Frontend development).

## Quick Start

```bash
# Launch the secure local web dashboard (loopback only)
uv run pilot dashboard
```

This starts a loopback-only FastAPI service on an ephemeral port and opens a one-time authenticated browser session.

### Alternative: Activate the virtual environment

```bash
source .venv/bin/activate
pilot dashboard
```

## Configuration

Credentials stay in `.env` (never committed, never exported to the browser, diagnostics, or backups as raw values).

```bash
# Tastytrade OAuth (account settings → API)
export TASTYTRADE_CLIENT_SECRET=your_client_secret
export TASTYTRADE_REFRESH_TOKEN=your_refresh_token

# Optional market/news providers
export MASSIVE_API_KEY=...
export BENZINGA_API_KEY=...
```

### Recommendations (Codex CLI / ChatGPT subscription)

AI recommendations use the **local Codex CLI**, authenticated through ChatGPT subscription sign-in (Codex CLI OAuth). Position Pilot never inspects, copies, or persists Codex OAuth tokens.

- Install and sign in to the Codex CLI separately using its supported ChatGPT flow.
- There is **no Anthropic / API-key recommendation path** in production.
- Core portfolio data remains available when Codex is signed out or unavailable.

Application settings and snapshots live under `~/.local/share/position-pilot/` (override with `POSITION_PILOT_DATA_DIR`). Market data cache: `~/.cache/position-pilot/` (~10 minutes).

The app verifies that `.env` is gitignored and warns if it appears tracked or broadly readable. Credential values are never read into diagnostic payloads.

## Commands

```bash
# Web dashboard (only dashboard surface)
pilot dashboard
pilot dashboard --no-browser

# Positions (grouped by strategy)
pilot positions
pilot positions --raw

# Quote
pilot quote SPY

# Deterministic health metrics (Codex recommendations: use the web UI)
pilot analyze

# Market overview
pilot market

# Watchlist
pilot watchlist show
pilot watchlist add TSLA
pilot watchlist remove SPY

# Accounts
pilot account list
pilot account set 12345678
```

## Architecture

```
src/position_pilot/
├── cli.py                 # Typer CLI
├── client/tastytrade.py   # Read-only Tastytrade API client
├── domain/                # Application services (portfolio, risk, catalysts, recommendations, operations, …)
├── providers/             # Tastytrade, Codex CLI, Massive, Benzinga
├── persistence/sqlite.py  # Versioned SQLite + backups
├── streaming/             # DXLink + account streaming
├── analysis/              # Strategy detection, market analytics
└── web/                   # FastAPI + prebuilt static frontend
frontend/                  # React / TypeScript / Vite sources
```

### Key design

1. **Web-only dashboard** — legacy Textual TUI removed in Phase 7.
2. **LLM recommendations via local Codex CLI** — ChatGPT subscription; no Anthropic SDK.
3. **Domain services** shared by CLI and web.
4. **Deterministic risk** never depends on AI.
5. **Read-only, loopback-only** security model.
6. **Operations** — CSV/HTML/PDF export, redacted diagnostics, retention controls, backup/restore, explicit update readiness (never auto-install).

Confirmed requirements: [docs/PRD.md](docs/PRD.md), [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

### Data flow

```
Tastytrade API → PortfolioService snapshot
                  ↓
detect_strategies() → multi-leg groups (never cross-account)
                  ↓
RiskService / CatalystService / RecommendationService (Codex)
                  ↓
Web dashboard (decision support only)
```

## Operations (Settings → Diagnostics)

From the web Settings panel (authenticated local session):

| Capability | Notes |
|------------|--------|
| Portfolio / history CSV | Current snapshot and historical summaries |
| Printable HTML / PDF | Timestamped, attributed, decision-support disclaimer |
| Redacted diagnostic JSON/ZIP | No credentials, tokens, cookies, prompts, licensed full text, or raw env |
| Retention | Preview then explicit apply; audit-critical history not silently purged |
| Backups | List / create / download / restore with integrity checks; restore blocked while monitoring runs; pre-restore backup always created |
| Update readiness | Current/schema awareness, backup required, never auto-installs or runs package managers |
| `.env` checks | gitignore / tracking / permissions warnings without reading values |

## Migration and updates

1. Disable monitoring if it is running.
2. Create a manual backup from Settings.
3. Install the new version yourself (`git pull` + `uv sync --group dev` as appropriate).
4. Launch `pilot dashboard` and confirm schema migrations complete.
5. Restore the pre-update backup if needed.

Backups occur before migrations and daily while monitoring runs (seven daily / four weekly). `.env` is excluded.

## Frontend development

Supported contributor toolchain:

- **Node.js** 22.x (see `frontend/.nvmrc`)
- **pnpm** 11.7.0 (see `packageManager` in `frontend/package.json`)

```bash
cd frontend
pnpm install
pnpm run typecheck
pnpm run build          # writes into src/position_pilot/web/static/
pnpm run test:browser   # Playwright + axe
```

## Testing

```bash
uv sync --group dev
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
uv run pytest tests/test_operations.py -q

# Opt-in live Tastytrade read-only smoke (accounts/positions/quotes only)
POSITION_PILOT_LIVE_SMOKE=1 uv run pytest tests/test_live_tastytrade_smoke.py -q

cd frontend && pnpm run typecheck && pnpm run build && pnpm run test:browser
uv build
```

## Scale targets

Architecture targets (regression-tested): 5 accounts, 500 open legs, 200 strategies, 100 watchlist symbols — via batch enrichment, shared services, and frontend table pagination.

## Technology stack

- Python 3.12+, FastAPI, Typer, Pydantic, Rich, httpx, uvicorn
- React, TypeScript, Vite, pnpm
- Local Codex CLI for AI recommendations (ChatGPT subscription OAuth)
- Tastytrade OAuth for read-only market data and positions

## Requirements

- Python 3.12+
- Tastytrade account (OAuth client secret + refresh token)
- Optional: Codex CLI signed in with ChatGPT for recommendations
- Optional: Massive / Benzinga keys for extended market/news

## License

Apache 2.0
