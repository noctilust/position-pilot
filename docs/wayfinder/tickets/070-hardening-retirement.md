---
status: closed
type: task
blocked_by:
  - Deliver full portfolio feature parity
  - Replace Anthropic with Codex monitoring
---

# Harden the web app and retire the TUI

## Question

What parity, accessibility, performance, migration, export, backup, live-smoke, and regression evidence is required before removing the Textual dashboard?

## Scope (Phase 7 exit)

- Web-only dashboard: remove Textual TUI and `--tui`; remove Anthropic SDK and legacy Claude analyzer/recommendation cache
- Keep useful read-only CLI; `pilot analyze` uses deterministic risk metrics; Codex recommendations via web / local Codex provider
- WCAG 2.2 AA practical automated coverage: axe, keyboard, Escape/focus, reduced-motion, narrow + large responsive checks, deterministic visual contract
- Operations domain/API/UI: CSV export, redacted diagnostics, HTML/PDF snapshots, retention preview/apply, backup list/create/download/restore, update readiness, `.env` diagnostics
- Scale targets: ≥5 accounts, 500 legs, 200 strategies, 100 watchlist symbols with deterministic tests and frontend pagination
- Calculation regression suite + opt-in live Tastytrade read-only smoke (`POSITION_PILOT_LIVE_SMOKE=1`)
- Packaging: prebuilt frontend in wheel force-include; Node 22 + pnpm 11.7 documented; migration/backup/update guidance; schema migration compatibility preserved

## Resolution

### Web-only retirement

- Deleted `src/position_pilot/dashboard/` (Textual app + widgets)
- Removed `analysis/llm_signals.py` and `analysis/recommendation_cache.py`
- Dropped `anthropic` and `textual` runtime dependencies from `pyproject.toml`
- CLI `dashboard` launches web only; `--tui` rejected
- CLI `analyze` uses `PortfolioService` + `RiskService` (no Anthropic); points traders to web for Codex recommendations
- README and this ticket updated for ChatGPT/Codex CLI OAuth, Tastytrade credentials, web-only dashboard

### Operations domain (`domain/operations.py`)

Cohesive service (not route-scattered) for:

| Capability | Behavior |
|------------|----------|
| CSV export | Current portfolio + history summaries; decision-support disclaimer |
| Diagnostics | **Strict allowlist** redacted JSON + ZIP; path always logical `.env` (never absolute home paths); no account/default_account/account_number, raw broker numbers, local AI identity, credentials, tokens, cookies, prompts, licensed full text, or raw env |
| Printable snapshot | Timestamped HTML + multi-page valid PDF (`%PDF` … `%%EOF`, MediaBox 612×792) including **all** strategies with provider attribution, as-of, schema, disclaimer |
| Retention | Settings + truthful preview + explicit `confirm=true` apply; **no** `clear_confirmed` mode; ordinary apply **never** clears roll chains / trader decisions / audit events / recommendation history / transactions; older 30-minute snapshots compact to one durable daily summary per day after the retention window |
| Backups | Server-local full SQLite backups remain faithful for restore; browser download is a **sanitized portable ZIP** (pseudonymized broker numbers, `full_text` removed, credentials/tokens/prompts/cookies excluded) with sentinel tests; list uses basename/id only; unified sidecar uses `__version__` + schema/integrity/exclusions |
| Restore / update guards | **Fail closed** on monitoring unknown/error; reject restore when backup schema > `CURRENT_SCHEMA_VERSION` before live replacement; pre-restore backup + atomic replace |
| Update readiness | Current version/schema, backup required, monitoring guard, reversible instructions; `auto_install=false`; never executes package managers |
| `.env` diagnostics | gitignore/tracking/permission warnings; never reads values; path field is `.env` |

Authenticated versioned APIs under `/api/v1/exports/*`, `/api/v1/diagnostics/*`, `/api/v1/settings/retention*`, `/api/v1/backups*`, `/api/v1/update/status`. Settings UI surfaces confirmation and error states.

### Accessibility / UX

- Focus-visible on controls; skip link; drawer focus trap + Escape restore
- `prefers-reduced-motion` honored (including spin)
- Responsive rules for narrow (≤48rem) and large (≥90rem) viewports
- **Phone (390 CSS px / ≤48rem):** simplified **read-only** layout — editing, destructive, and recommendation actions absent/disabled; accessible `role="status"` notice; portfolio / alerts / catalysts remain useful; desktop controls return at wide widths
- Positions/orders tables windowed (50/40 page sizes) with keyboard pager controls
- Playwright: axe on shell/drawer/settings; keyboard; reduced-motion; 390px + 1600px; structural visual contract + **`toHaveScreenshot` baseline** (`overview-workspace.png`)

### Scale / validation

- `tests/test_scale_performance.py` — 5 accounts / 500 legs / **exactly 200** strategy groups / **exactly 100** watchlist symbols with **bounded batch** provider calls (1 quotes batch + 1 metrics batch)
- `tests/test_calculation_regression.py` — iron condor / vertical / equity isolation
- `tests/test_live_tastytrade_smoke.py` — opt-in read-only; default skip
- `tests/test_operations.py` — diagnostics allowlist, portable backup sentinels, retention compaction, fail-closed restore/update, multi-page PDF 200 strategies, schema-newer reject
- `tests/test_packaging_sdist.py` — sdist excludes `node_modules`/caches/Playwright artifacts; wheel includes static frontend

### Packaging

- Wheel ships prebuilt `web/static`
- Sdist includes frontend sources/config but **excludes** `frontend/node_modules`, caches, Playwright reports/results
- `frontend/.nvmrc` → Node 22; `packageManager: pnpm@11.7.0`
- Schema version remains migration-compatible (`CURRENT_SCHEMA_VERSION` unchanged; daily snapshot compaction reuses `portfolio_snapshots` with `state='daily'`)

## Exit evidence (verified)

```bash
uv sync --group dev
uv run ruff check src/position_pilot/domain/operations.py \
  src/position_pilot/domain/market.py src/position_pilot/domain/snapshots.py \
  src/position_pilot/persistence/sqlite.py tests/test_operations.py \
  tests/test_market_service.py
uv run pytest                          # 183 passed, 1 skipped (live smoke)
cd frontend && pnpm run typecheck && pnpm run build && pnpm run test:browser  # 5 passed
uv build                               # sdist + wheel; no node_modules in archives; wheel has static
```

### Phase 7 repair pass 2 (audited blockers)

1. **Provider batch fallback** — `ProviderRoutedMarketSource.get_quotes_batch` batches Tastytrade primary first, then resolves every missing/unusable symbol via FieldRouter/`get_quote` (Massive provenance preserved). Healthy 100-symbol primary stays one batch.
2. **Retention compaction** — daily rows store fully valid `PortfolioSnapshot` payloads (`state=daily` + `SnapshotCompaction` metadata). `list_portfolio_snapshot_payloads` and `export_history_csv` work without Pydantic failures and retain NLV/P&L/accounts/strategies.
3. **Restore schema authority** — SQLite `schema_migrations` is authoritative; future schemas rejected; sidecar/database mismatch rejected (tested with future DB + stale sidecar).
4. **Phone read-only @ 390px** — Settings hides recommendation/catalyst forms, Create backup, retention apply/save, restore, monitoring toggles; desktop controls return at wide width.
5. **HTML printable attribution** — includes `snapshot.freshness.provider` and as-of timestamp alongside generated/captured/schema/disclaimer.

## Honest limitations

- PDF is a minimal multi-page Helvetica text PDF (not a full print layout engine).
- Update “latest version” is not auto-probed from the network (no package-manager execution); remains null unless a probe is injected.
- Full WCAG 2.2 AA is covered by practical automated axe + keyboard/responsive contracts, not a manual audit of every control state.
- Scale budgets are soft CI ceilings, not hard real-time SLOs on production hardware.
- Live Tastytrade smoke requires user-configured credentials and explicit env flag.
- Portable browser backups are intentionally lossy (sanitized) and must not be used for server restore.
