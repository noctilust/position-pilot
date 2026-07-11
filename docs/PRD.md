# Position Pilot Web Revamp PRD

**Status:** Confirmed
**Product mode:** Local, read-only trading workstation
**Primary user:** An active Tastytrade options trader managing strategic holdings and tactical strategies across multiple accounts

## 1. Product outcome

Position Pilot becomes a local web application that helps a trader understand portfolio risk, react quickly to tactical option positions, and connect unusual stock or option movement to credible evidence.

The product must remain explicitly read-only. It may analyze, model, notify, and recommend, but it must never create, stage, submit, replace, or cancel an order.

## 2. Success criteria

- Cached dashboard content is usable within 3 seconds.
- A live portfolio refresh completes within 10 seconds under normal provider conditions.
- New held-symbol catalysts appear within 5 minutes of provider publication.
- Every catalyst contains a source, timestamp, confidence, attribution level, and evidence summary.
- No uncited catalyst claim is presented as fact.
- `No confirmed catalyst found` is a valid, visible result.
- Provider failure never blocks access to the most recent portfolio snapshot.
- Account numbers, secrets, and personal identifiers never reach the browser or AI prompts.
- Strategy grouping, P/L, Greeks, and roll calculations remain regression-compatible with the existing application.
- Every current user-facing capability has a web equivalent before the TUI is removed.

## 3. Experience architecture

The primary interface is a local web dashboard launched with `pilot dashboard` and bound to `127.0.0.1`. The Python CLI remains available for quick commands and automation.

Primary areas:

1. **Overview** — account health, buying power, risk summary, major movers, catalysts, and alerts.
2. **Positions** — strategies, legs, Greeks, P/L, catalyst state, horizon, and AI actions.
3. **Position detail** — charts, legs, breakevens, risk, news, option-market drivers, recommendations, trade plan or thesis, and roll chain.
4. **Roll analytics** — history, patterns, heatmap, chain credits, and manual corrections.
5. **Markets** — watchlist, quotes, IV environment, events, and broad-market context.
6. **Alert Center** — deduplicated critical, action, and informational alerts.
7. **Settings** — accounts, providers, cadence, thresholds, notifications, privacy, retention, and diagnostics.

The application supports a saved primary account, fast account switching, and an `All Accounts` view. Legs from different accounts must never be grouped into one strategy.

## 4. Visual and interaction direction

The interface is a calm, information-dense professional trading workstation.

- Dark mode is the default; light mode is fully supported.
- Risk is never conveyed by red/green alone.
- Numeric data uses tabular figures.
- Dense tables progressively become structured cards on small screens.
- Desktop is optimized from 1280px upward; tablet remains fully usable; phone uses a simplified read-only layout.
- WCAG 2.2 AA, complete keyboard navigation, visible focus, reduced motion, semantic tables, and screen-reader labels are required.
- The visual language favors precise dividers, strong typography, asymmetric information hierarchy, and restrained semantic color over generic card grids, gradients, glass effects, or decorative charts.

## 5. Portfolio and position model

Positions are assigned an editable horizon:

- `Strategic` by default for standalone stock and long LEAPS.
- `Tactical` by default for multi-leg and short-dated options.
- Protective positions inherit the linked holding's horizon.
- Ambiguous positions remain `Unclassified` until confirmed.

Strategic holdings support a trader-authored thesis: purpose, expected duration, target range, invalidation conditions, income or hedge intent, and events to watch.

Tactical strategies support a trade plan: entry thesis, intended duration, profit target, maximum acceptable loss, roll criteria, event exposure, and exit deadline.

AI recommendations are produced at the strategy-within-account level. Standalone equities receive individual recommendations. A separate portfolio-level assessment summarizes concentration and correlated exposure.

## 6. Deterministic risk engine

Objective calculations must not depend on AI:

- Max profit, max loss, and breakevens where defined.
- Combined delta, gamma, theta, vega, IV, DTE, and strike distance.
- Concentration, account and portfolio exposure, liquidity, and event exposure.
- Chain-adjusted and current-position P/L.
- Standard price, IV, theta, and expiration stress scenarios.

If AI is unavailable, the application continues to show measurements and triggered conditions without translating them into trading advice.

## 7. Catalyst intelligence

Every unique held underlying is scanned. Quiet symbols remain visually subdued; meaningful movers and high-impact events are promoted.

Default meaningful-move thresholds:

- Individual stock: absolute move of at least 2% from official prior close.
- Broad-market ETF: absolute move of at least 1%.
- A move unusually large relative to recent intraday behavior.
- Any confirmed high-impact event, even before a large move develops.

Catalyst confidence:

- `Confirmed` — primary evidence or reliable reporting directly connects the event and security.
- `Likely` — timing and relevance are strong, but causation is not explicit.
- `No confirmed catalyst found` — evidence is insufficient.

Attribution level:

- Company-specific.
- Industry or peer-driven.
- Macro or market-wide.
- Options-market driver.
- No confirmed catalyst.

The default lookup window begins at the previous market close and continues through the current moment, including extended hours. It may extend to 72 hours for major scheduled events still affecting price discovery.

Trusted evidence is prioritized as company releases and filings, regulators and government sources, established financial reporting, and reputable specialist publications. Social sentiment may appear only as a separately labeled, low-confidence side note and never as catalyst evidence.

The application must also explain options-market movement when the underlying is quiet, including IV expansion or crush, skew, unusual volume, liquidity, dividend effects, earnings proximity, expiration, and gamma risk. These mechanisms must not be mislabeled as news.

## 8. Provider policy

Provider priority is field-specific and provenance is retained on every derived record.

1. Tastytrade is authoritative for accounts, balances, positions, transactions, orders, held-contract marks, live Greeks, and available market data.
2. Massive Stocks/News fills historical-bar, event, filing, and baseline-news gaps.
3. Massive Options fills missing historical option bars, quotes, volume, open interest, IV, and chain data.
4. Benzinga or another licensed feed provides premium real-time catalyst coverage where configured.
5. Public web sources may supplement evidence only without bypassing authentication, paywalls, CAPTCHAs, robots restrictions, access controls, or anti-bot mechanisms.

Values from different providers are never silently averaged. Material discrepancies are visible in diagnostics.

Full article text may be retained only when the active provider agreement permits storage and AI processing. Provider removal notices must be honored. Otherwise the application stores permitted metadata, excerpts, derived catalyst records, and source URLs.

## 9. AI provider and privacy

Anthropic is replaced by a local Codex provider. Position Pilot invokes the installed Codex CLI, which is authenticated through its supported ChatGPT sign-in flow. The application must never inspect, copy, or persist Codex OAuth tokens.

AI output uses a versioned structured schema. Prompts contain only the minimum analytical context: symbol, strategy, quantities, prices, P/L, Greeks, DTE, volatility, relevant catalysts, plan or thesis, and aggregated exposure. They exclude account numbers, names, credentials, transfers, and unrelated positions.

When Codex is signed out, rate-limited, or unavailable, AI actions fail transparently while the rest of the dashboard continues working. An API-key provider may exist only as an explicitly enabled fallback; the application never switches providers silently.

## 10. Monitoring and scheduling

Trading-day monitoring runs from 7:30 AM through 6:00 PM America/New_York, adjusted for exchange holidays and early closes.

- Market and tactical risk inputs are monitored every 60 seconds.
- A deterministic reevaluation runs every 30 minutes.
- Tactical AI is invoked when material inputs change or an event trigger fires.
- Strategic AI runs daily or on a material event.
- Unchanged fingerprints update `last_evaluated_at` without consuming a Codex call.
- `last_evaluated_at` and `recommendation_updated_at` remain distinct.

A lightweight background monitor is enabled only after explicit onboarding consent. Missed intervals are not replayed. On wake or network recovery, one immediate evaluation runs when inside the monitoring window.

Material recommendation changes notify the trader only when action, urgency, or risk changes. Minor wording updates remain silent.

## 11. Streaming and reconciliation

Tastytrade streaming is used for DXLink market data and all available account, balance, position, order, and transaction events. REST snapshots reconcile state at startup, every 5 minutes, after reconnects, and after sequence gaps or impossible states.

Streaming provides responsiveness; versioned REST snapshots remain the reconciliation authority.

## 12. Persistence and retention

SQLite stores normalized settings, snapshots, provider records, catalyst events, alerts, theses, trade plans, recommendation history, and roll analytics.

Credentials remain in `.env`. The application verifies that `.env` is ignored and warns if it is tracked or broadly readable. Secrets are never logged, exported, or returned to the frontend.

Retention defaults:

- Transaction and roll-chain history: indefinite until manually cleared.
- Thirty-minute portfolio snapshots: one year, then compacted into daily summaries.
- Catalyst events and source links: one year.
- Article metadata and permitted excerpts: 90 days.
- Full article text: provider-license dependent.

Backups occur before every migration and daily while monitoring runs, retaining seven daily and four weekly copies. `.env` is excluded.

## 13. Offline and degraded operation

The dashboard remains usable from cached data. A persistent notice states that the network is unreachable and identifies the exact cached snapshot timestamp. Portfolio, market, news, and AI panels retain independent freshness timestamps.

Each provider degrades independently. Stale data is labeled and never reused to explain a new price move without an incomplete-coverage warning.

Core portfolio panels switch atomically between versioned snapshots. News and AI may update independently afterward.

## 14. Alerts and notifications

The Alert Center groups risk, catalyst, recommendation, and provider-health events with severity, account, underlying, strategy, source, timestamp, and resolution state. It supports acknowledge, snooze, and mute-by-rule.

macOS notifications are privacy-safe by default, containing symbol, strategy, and alert type but no account numbers, quantities, balances, or P/L. Rich previews are opt-in.

## 15. History, feedback, and auditability

Recommendation history is immutable and presented as diffs in action, urgency, risk, evidence, catalysts, and inputs. Unchanged checks collapse into daily summaries.

Trader decisions are recorded separately as `Accepted`, `Dismissed`, `Deferred`, or `Handled in Tastytrade`, with optional notes.

Catalyst feedback supports `Relevant`, `Not related`, and `Missing catalyst`. Feedback improves local ranking but never silently rewrites historical evidence or trains an external model.

Roll detection preserves raw broker data and supports audited link, unlink, split, and merge corrections.

## 16. Packaging, exports, and updates

The React frontend is prebuilt into the Python package so normal users need only Python and `uv`. Contributors use a pinned Node version and pnpm.

Exports include CSV for portfolio and history data, redacted JSON for settings and diagnostics, and printable HTML/PDF snapshots with timestamps and attribution. Credentials and licensed full text are excluded by default.

Updates are explicit, versioned, migration-aware, backed up, and reversible. They are never installed automatically while monitoring is active.

## 17. Scale target

The architecture must support five accounts, 500 open legs, 200 detected strategies, and 100 watched symbols through batching, shared per-underlying enrichment, prioritization, and table virtualization.

## 18. First-release non-goals

- Brokerage writes or automated trading.
- Cloud hosting or remote access.
- Multi-user collaboration.
- Native mobile applications.
- Social sentiment as catalyst evidence.
- Bypassing protected web content.
- Monte Carlo or probabilistic backtesting.
