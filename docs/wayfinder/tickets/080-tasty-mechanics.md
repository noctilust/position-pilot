---
status: open
type: feature
blocked_by: []
---

# Tasty Mechanics vertical slice (v1)

## Question

How should Position Pilot combine versioned tastylive educational mechanics with read-only tastytrade position facts, deterministic rule evaluation, risk-gated advisory candidates, shadow-mode UI, and constrained Codex context—without ever placing orders?

## Product separation

| Layer | Source of truth | Role |
|-------|-----------------|------|
| **Facts** | tastytrade-derived snapshots, RiskService, market context, rolls, NLV/BP | Deterministic, never fabricated |
| **Policy / playbook** | Versioned built-in playbooks (`tastylive-short-premium.v1`) | Educational presets + official source metadata only |
| **Trader plan** | Thesis / trade plan documents | Overrides playbook defaults when parseable |
| **Codex** | Local CLI recommendation | Explains/compares supplied candidates only |

Priority: **safety/data-integrity gates > trader plan/override > playbook defaults > Codex explanation**.

## Playbook versioning

- Playbook id: `tastylive-short-premium.v1` (id embeds version; `version` field is `v1`).
- Supported short-premium set excludes detector `Butterfly` (debit long-wing structure) unless a future version can prove net-credit economics.
- Sources store: stable `source_id`, title, URL, optional publication date, `reviewed_at`, and citing `rule_ids`.
- **Never** fetch source URLs at runtime; **never** store article bodies.
- To add/review a playbook version:
  1. Copy the built-in playbook factory in `domain/mechanics.py` (or add `v2` factory).
  2. Assign a new `playbook_id` (e.g. `tastylive-short-premium.v2`).
  3. Update source `reviewed_at` and rule citations.
  4. Extend `MechanicsSettings` validator / `get_playbook` allowlist.
  5. Add offline regression fixtures for new rules and stable IDs.
  6. Document educational disclaimer: defaults are not guarantees.

## Rules (v1)

- `gate.strategy_supported` — fail closed outside short-premium set
- `gate.data_quality` — stale/wide/partial history blocks
- `gate.event_exposure` — high-impact catalyst watch
- `gate.assignment_expiration` — near-expiry review
- `gate.manual_execution` — always advisory
- `profit.manage_winner` — default ~50% of known original credit
- `time.manage_at_dte` — default manage-at 21 DTE (stronger for undefined risk)
- `size.small_position` — defined max-loss vs NLV; undefined BPR not invented
- `tested.side_review` / `roll.untested_side` / `roll.credit_only`
- `risk.defined_vs_undefined`

## Candidates

Kinds: `hold`, `close`, `reduce`, `roll-review`, `manual-review`. **Never `ADD`.**
Roll review without a quoted chain leaves economics/after-risk **unknown**. No broker order construction.

## Settings (SQLite `mechanics`)

- `enabled`, `shadow_mode` (default **true**), `playbook_id`
- `profit_target_pct`, `manage_at_dte`, `tested_delta_threshold`
- `defined_risk_cap_pct`, `undefined_bpr_cap_pct`, `credit_only_rolls`
- Malformed settings fail closed to safe defaults; `advisory_only` is always true.

## Shadow mode

- Default on: UI/API show mechanics; Codex receives compact allowlisted context.
- Does **not** emit mechanics-specific notifications or override recommendation `action`.
- Mechanics fingerprint still joins strategy recommendation input fingerprint so rule crossings can re-due evaluation.

## Data gaps

Missing opening credit, BPR, quotes, or partial roll history → explicit `unknown` / `blocked` / `manual-review`. Never invent credits, margins, strikes, or roll quotes.

- **Catalysts:** `catalysts=None` means availability unknown (WATCH); `catalysts=[]` means known empty (PASS/no-high-impact). Never convert load failures to known-empty.
- **Sizing:** Undefined-risk BPR is never proxied from market value or account remaining buying power. Market-value/NLV is informational only.
- **Liquidity:** `underlying_spread_pct` is underlying quote quality only. Option/complex liquidity is `option_liquidity_known=false` until leg/complex quotes exist.
- **Timestamps:** Malformed or timezone-naive market `as_of` → `invalid_market_timestamp` (fail closed).
- **Codex:** Outside shadow mode, candidate-bearing recommendations are post-validated; incompatible `add`/`hedge`/`roll` fail closed as `invalid_output`. Shadow mode remains observational.

## Manual execution boundary

Position Pilot remains **incapable** of create/stage/dry-run/submit/replace/cancel of broker orders. UI states execution remains manual in tastytrade.

## Surfaces

- Domain: `MechanicsService`, pure `evaluate_mechanics` / `replay` offline harness
- API: `GET/PUT /api/v1/settings/mechanics`, `GET /api/v1/strategies/{id}/mechanics`, strategy detail embeds `mechanics`
- UI: strategy detail “Tasty mechanics” panel (shadow/advisory labels, rules, candidates, sources)
- Codex: allowlisted `mechanics` on strategy context + prompt constraints when candidates present

## Exit criteria

- Offline evaluation returns versioned rules + candidates
- Missing facts never fabricated
- Recommendations receive sanitized mechanics without breaking portfolio/equity paths
- Browser can inspect sources and shadow label
- No order mutation path
- Tests: unit, API, recommendation context, browser as applicable
