---
status: closed
type: prototype
blocked_by:
  - Extract domain services and versioned snapshots
  - Build evidence-backed catalyst intelligence
---

# Replace Anthropic with Codex monitoring

## Question

How should the Codex CLI provider, structured outputs, horizon-aware prompts, fingerprints, tactical triggers, scheduler, notifications, and failure states work without token access?

## Scope

Phase 6 exit requires:

- Local installed Codex CLI subprocess provider (ChatGPT sign-in), never reading/copying/logging/persisting OAuth tokens
- Isolated temp cwd for Codex exec (schema/output only); prompt on stdin only
- Versioned Pydantic structured-output schema with strict validation
- Domain-boundary prompt minimization with catalyst/thesis/plan/exposure allowlists and recursive redaction
- Explicit signed-out / rate-limited / unavailable / invalid-output statuses; no silent fallback
- API-key fallback is not a functional/selectable path (disabled internal abstraction only)
- Shared domain models for strategy-within-account, standalone equities, and portfolio assessment
- Horizon defaults: equities/LEAPS strategic (daily/material inputs); multi-leg options tactical
- Distinct `last_evaluated_at` vs `recommendation_updated_at`
- Deterministic fingerprints; unchanged inputs skip Codex unless `force=True` (material_event is scheduling-only)
- Immutable recommendation history with audit diffs (reasoning/evidence/catalysts/inputs) and notification-only materiality for action/urgency/risk
- Separate immutable trader decisions
- Monitoring window 7:30–18:00 ET (early-close market 13:00 → monitor until 15:00), deterministic US calendar, 60s tactical pulse, 30m reevaluation
- No missed-interval replay; wake via scheduler time-gap; recovery via portfolio unavailable→available
- Consent-gated monitoring; single-flight cycles; per-subject isolation
- Risk + catalyst + recommendation + provider-health alerts with stable dedupe
- Privacy-safe macOS notifications (symbol · strategy · alert type by default)

## Resolution (post-repair)

- **Codex provider** — temp cwd isolation; stdin prompt; auth TTL cache; signed_out vs timeout/unavailable; fail-closed unknown auth; bounded retry for unavailable/rate_limit only
- **Recommendations** — allowlisted/redacted contexts; fingerprint skip unless force; audit vs notification materiality; per-subject locks; atomic recommendation+history persistence; concurrency-safe daily summaries
- **Monitoring** — algorithmic US holidays/early closes for any year; 15:00 early-close monitor end; risk_tick evaluates tactical fingerprint changes; wake/recovery end-to-end without OS listeners; equity dedupe vs Long/Short Stock; exception isolation; restored runtime timestamps
- **Alerts** — risk + high-impact catalyst + recommendation + provider-health with dedupe
- **Web/Frontend** — threadpooled bootstrap/status; no selectable API-key fallback; history diffs in drawer; Phase 6 UI labels
- **TUI / Anthropic** — retained until Phase 7

## Exit evidence (second repair)

- Account-id sanitizer no longer redacts strike chains/prices (`500/505/480/475` preserved; broker shapes like `5WT00001` redacted)
- Risk alerts use discrete states (normal/elevated/high/critical) with threshold-crossing only; small normal P/L noise emits zero alerts
- 60s pulse uses injectable live market context (DXLink hub preferred, MarketService fallback); material price/IV/spread thresholds gate Codex; penny ticks observe without calls
- `tick_once` async scheduler seam covers wake gap without multi-interval replay
- Codex exec adds `--ignore-user-config` and `--ignore-rules` while keeping stdin/temp-cwd/schema isolation
- Full pytest: **156 passed**; scoped Ruff clean; frontend typecheck/build clean
- Python sdist/wheel build clean; Playwright smoke/accessibility: **1 passed**

## Honest limitations

- Live Codex CLI structured-output behavior depends on installed CLI version; tests mock subprocesses.
- Ad-hoc exchange emergency closures (weather, national mourning) are not modeled — algorithm covers standard NYSE holiday/early-close rules only.
- Wake detection uses scheduler `tick_once` time-gap heuristics, not OS sleep/wake APIs.
- Network recovery uses portfolio loader unavailable→available transitions, not OS network reachability APIs.
- Live DXLink market context is best-effort when streaming hub is active; otherwise MarketService refresh is used.
- Material market thresholds are deterministic defaults (0.5% price, 2 IV-rank points, 0.02 IV, 0.5pp spread), measured from the last material baseline, and may need trader tuning later.
- API-key recommendation provider remains intentionally non-functional (not selectable).
- Anthropic remains a runtime dependency for the legacy TUI until Phase 7.
- macOS notifications require `osascript` and user notification permissions.
