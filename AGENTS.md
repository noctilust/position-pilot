# AGENTS.md

Shared repository policy for all agents working in this repo.

## Agent-specific instructions

| Agent | Instruction artifact |
|-------|----------------------|
| **Codex** (orchestration, review, integration) | [`.codex/skills/position-pilot-codex-orchestrator/SKILL.md`](.codex/skills/position-pilot-codex-orchestrator/SKILL.md) |
| **Grok Build** (implementation, testing, repair) | [`.grok/position-pilot-implementation.md`](.grok/position-pilot-implementation.md) |

**Workflow model:** Codex defines the contract; Grok owns the implementation loop; Codex independently reviews and integrates. Prefer Grok Build subscription credits first, then Codex subscription credits. Never consume OpenAI or xAI API credits.

---

## Billing and authentication

Use only locally authenticated **subscription-backed** CLIs.

**Allowed:** Codex CLI (ChatGPT subscription), Grok Build CLI (Grok subscription), local shell, local Git, local test/build/lint/format/analysis tools.

**Prohibited:**

- `OPENAI_API_KEY`, `XAI_API_KEY`
- OpenAI or xAI API calls
- Configuring xAI as a Codex model provider, or OpenAI as a Grok model provider
- Silent fallback from subscription auth to API-key auth
- Any other third-party API that incurs charges without explicit user approval

When invoking Grok, strip API keys when practical:

```bash
env -u XAI_API_KEY -u OPENAI_API_KEY grok ...
```

Do not modify the user's shell profile or permanently unset environment variables unless asked.

Credit priority: (1) Grok Build subscription → (2) Codex subscription → (3) stop and report before any paid API.

### App runtime credentials (not agent billing)

Configured app credentials (`TASTYTRADE_*`, `MASSIVE_API_KEY`, `BENZINGA_API_KEY`, and similar) are product runtime secrets, not agent-tooling auth.

- Prefer mocks, fixtures, and recorded responses when they suffice for implementation and tests.
- Use live broker/LLM paths only when the task explicitly requires them **and** the user has already configured those env vars.
- Do not run broad live LLM portfolio analysis “for validation” without need.

---

## Instruction precedence

1. Explicit user instruction in the current task
2. Security, privacy, and data-integrity requirements
3. Nearest applicable nested `AGENTS.md`
4. This root `AGENTS.md`
5. Existing repository conventions
6. Agent defaults

A nested `AGENTS.md` applies to its directory and descendants. Credit policy never overrides local coding conventions or user requirements. Repo tooling conventions in this file beat global agent defaults (for example: this repo uses `pnpm`, not bun/npm, for frontend).

---

## Git safety

**Never** without explicit user authorization:

- Discard, reset, clean, restore, or overwrite user-authored changes
- Rewrite, rebase, or amend user commits
- Force-push
- Delete branches/worktrees unless they are agent-created and confirmed disposable
- Push, create PRs, merge protected branches, deploy, publish, modify cloud/remote resources, or trigger production workflows

Prefer agent branches `grok/<task-id>` and worktrees `<repository>-grok-<task-id>`. Do not require commits unless useful for isolation/integration. Commits must be descriptive, task-scoped only, and not amend unrelated history.

---

## Quality standards

- Satisfy stated requirements; preserve behavior outside scope
- Follow repository conventions; prefer the smallest coherent change (not brittle hacks)
- Minimize complexity; avoid speculative abstractions and unrelated cleanup
- Handle important error paths; avoid silent data corruption and secret exposure
- Include appropriate tests; remain understandable to maintainers

---

## Security and privacy

Do not: expose credentials; print secrets unnecessarily; commit `.env` with secrets; transmit private repo content to unrelated services; add telemetry without approval; weaken auth to pass tests; disable certificate validation without a documented requirement; run untrusted scripts carelessly; use production credentials for testing.

Minimize external network access. Subscription auth is not permission to send private content to arbitrary external systems.

Product constraints:

- **Read-only:** never create, stage, submit, replace, or cancel broker orders.
- Bind the web service to loopback only unless the user explicitly requires otherwise.
- Do not send account numbers, secrets, or personal identifiers to the browser or AI prompts.

---

## Dependencies

Do not add a dependency when the repo or standard library already solves the problem reasonably. Assess necessity, maintenance, license, security, size, runtime cost, and environment compatibility. No broad upgrades unless required. Lockfile changes must match intentional dependency changes.

---

## Testing (shared)

- Validate behavior, not implementation details, where practical
- Do not weaken, delete, skip, or broadly mock tests just to pass
- Do not update snapshots/fixtures without confirming the new output is intended
- If a test cannot run, report the command, failure, environment vs code cause, and what remains unverified
- Do not claim tests passed unless the command completed successfully

---

## Parallelism

Do not run multiple write-capable agents in the same working tree. Parallel Grok workers only with separate worktrees, independent tasks, clear branch ownership, and managed conflict risk.

---

## Build and development

```bash
# Install (Python + dev deps)
uv sync --group dev

# Prefer uv run so an activated venv is not required
uv run pilot dashboard          # Local web dashboard (default)
uv run pilot positions          # Positions grouped by strategy
uv run pilot positions --raw    # Ungrouped
uv run pilot quote SPY
uv run pilot analyze            # Deterministic portfolio health metrics
uv run pilot market

# Backend checks
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
uv run pytest tests/test_web_app.py -q   # targeted example

# Frontend (from frontend/)
pnpm install
pnpm run typecheck
pnpm run build
pnpm run test:browser
```

**Tooling notes:**

- Backend packages via `uv` (not pip). Install with `uv sync --group dev` (there is no `uv install`).
- Frontend via **`pnpm`** under `frontend/` (not npm or bun). Lockfile: `frontend/pnpm-lock.yaml`.
- Playwright via existing `@playwright/test` config (`pnpm run test:browser` from `frontend/`).
- Never push `.env` or similar secret files.

---

## Architecture overview

Position Pilot is a local, **read-only** portfolio workstation for Tastytrade options traders. The loopback web dashboard (`pilot dashboard`) is the only dashboard surface; read-only CLI commands remain available for quick checks and automation.

**IMPORTANT:** Trading recommendations use the local Codex CLI authenticated through its supported ChatGPT subscription sign-in. The application never reads or persists Codex OAuth tokens, and deterministic portfolio risk does not depend on AI.

### Module structure

```
src/position_pilot/
├── cli.py              # Typer CLI entry point
├── config.py           # Config (~/.config/position-pilot/config.json)
├── cache.py            # Market-data cache helpers
├── client/
│   └── tastytrade.py   # Tastytrade API client (OAuth, positions, quotes, Greeks)
├── models/
│   ├── position.py     # Position, Account, Greeks, Recommendation
│   ├── roll.py         # Roll tracking models
│   └── transaction.py  # Transaction models
├── analysis/
│   ├── strategies.py   # Strategy detection (Iron Condors, Spreads, etc.)
│   ├── signals.py      # [DEPRECATED] Legacy rule-based signals
│   ├── market.py       # Market data & IV environment
│   └── roll_*.py       # Roll history / analytics / tracker
├── domain/             # Domain services (portfolio, risk, rolls, snapshots, etc.)
├── providers/          # External providers (Tastytrade, Codex, Massive, etc.)
├── streaming/          # Live market/account streaming
├── persistence/        # SQLite and durable local state
└── web/                # FastAPI app, launcher, static frontend bundle
```

Frontend sources live in `frontend/` (React, TypeScript, Vite) and build into `src/position_pilot/web/static/`. Prefer domain services over new ad-hoc paths when adding web/API behavior.

### Key design patterns

- **Singleton clients/services:** `get_client()`, `get_portfolio_service()`, `get_recommendation_service()`
- **Strategy detection:** groups raw legs into multi-leg strategies
- **Domain services:** portfolio, risk, rolls, snapshots, recommendations, etc.
- **Web UI:** loopback FastAPI backend plus React frontend
- **Codex recommendations:** local subscription-authenticated CLI; no API-key fallback

### Authentication (app runtime)

```
# Tastytrade OAuth
TASTYTRADE_CLIENT_SECRET
TASTYTRADE_REFRESH_TOKEN

# Optional market/news fallbacks
MASSIVE_API_KEY
BENZINGA_API_KEY

# Recommendations use local Codex CLI ChatGPT sign-in; no LLM API key
```

### Data flow

1. `TastytradeClient` fetches positions
2. `enrich_positions_greeks_batch()` adds Greeks and underlyings (10-min cache)
3. Intrinsic / extrinsic value calculated per contract
4. `detect_strategies()` groups multi-leg strategies
5. RiskService produces deterministic portfolio health metrics
6. RecommendationService invokes the local Codex CLI when requested or scheduled
7. Web dashboard and read-only CLI display results

### Cache

Market data cached ~10 minutes at `~/.cache/position-pilot/` (`get_quote`, `get_quotes_batch`, `get_market_metrics`). Bypass with `force_refresh=True` or dashboard refresh. Positions and account data are always fresh.

### Strategy types detected

Iron Condors, Iron Butterflies, Verticals, Straddles, Strangles, Calendars, Diagonals, Covered Calls, Protective Puts, Collars, Jade Lizards, Butterflies, Synthetic Longs/Shorts, Risk Reversals.

Synthetic and risk-reversal grouping does not require exact quantity matches; unit size follows the smaller leg.
