import {
  Activity,
  Bell,
  ChartNoAxesCombined,
  CircleDot,
  Clock3,
  Command,
  Gauge,
  Layers3,
  Moon,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sun,
  Unplug,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  acknowledgeAlert,
  currency,
  evaluateStrategyRecommendation,
  fetchAlerts,
  fetchCatalysts,
  fetchMarkets,
  fetchMonitoring,
  fetchOrders,
  fetchPortfolio,
  fetchPortfolioRisk,
  fetchRecommendations,
  fetchRollHeatmap,
  fetchRollPatterns,
  fetchRolls,
  fetchStrategyDetail,
  fetchStreamingState,
  fetchWatchlist,
  getBootstrap,
  muteAlerts,
  pct,
  recordTraderDecision,
  resolveAlert,
  runMonitoringEvaluation,
  saveCatalystSettings,
  saveMonitoringConsent,
  savePrimaryAccount,
  saveRecommendationSettings,
  saveStrategyHorizon,
  saveThesis,
  saveTradePlan,
  saveWatchlist,
  signed,
  snoozeAlert,
  submitCatalystFeedback,
} from "./api";
import { OperationsPanel } from "./OperationsPanel";
import type {
  AlertRecord,
  BootstrapPayload,
  CatalystScanSnapshot,
  CatalystSettings,
  LiveMarketTick,
  MarketOverview,
  MonitoringStatus,
  OrderRow,
  PortfolioAccount,
  PortfolioRisk,
  PortfolioSnapshot,
  RecommendationRecord,
  RecommendationSettings,
  RollChain,
  RollHeatmap,
  RollPatterns,
  StrategyDetail,
  SymbolCatalystResult,
  WatchlistSnapshot,
} from "./types";

const STRATEGY_PAGE_SIZE = 50;
const ORDER_PAGE_SIZE = 40;
/** Matches PRD phone simplified read-only layout (≤48rem / ~390 CSS px phones). */
const PHONE_MEDIA_QUERY = "(max-width: 48rem)";

function usePhoneLayout(): boolean {
  const [isPhone, setIsPhone] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia(PHONE_MEDIA_QUERY).matches : false,
  );
  useEffect(() => {
    const media = window.matchMedia(PHONE_MEDIA_QUERY);
    const onChange = () => setIsPhone(media.matches);
    onChange();
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return isPhone;
}

type AppState =
  | { kind: "loading"; message: string }
  | { kind: "ready"; payload: BootstrapPayload }
  | { kind: "error"; message: string };

type PortfolioState =
  | { kind: "loading" }
  | { kind: "ready"; snapshot: PortfolioSnapshot }
  | { kind: "error"; message: string };

const navigationIcons: Record<string, LucideIcon> = {
  Overview: Gauge,
  Positions: Layers3,
  "Roll analytics": ChartNoAxesCombined,
  Markets: Activity,
  Alerts: Bell,
  Settings: Settings2,
};

function App() {
  const isPhone = usePhoneLayout();
  const [state, setState] = useState<AppState>({
    kind: "loading",
    message: "Securing local session…",
  });
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [activeSection, setActiveSection] = useState("Overview");
  const [portfolio, setPortfolio] = useState<PortfolioState>({ kind: "loading" });
  const [accountOptions, setAccountOptions] = useState<PortfolioAccount[]>([]);
  const [risk, setRisk] = useState<PortfolioRisk | null>(null);
  const [markets, setMarkets] = useState<MarketOverview | null>(null);
  const [watchlist, setWatchlist] = useState<WatchlistSnapshot | null>(null);
  const [orders, setOrders] = useState<OrderRow[]>([]);
  const [rolls, setRolls] = useState<RollChain[]>([]);
  const [patterns, setPatterns] = useState<RollPatterns | null>(null);
  const [heatmap, setHeatmap] = useState<RollHeatmap | null>(null);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null);
  const [strategyDetail, setStrategyDetail] = useState<StrategyDetail | null>(null);
  const [catalysts, setCatalysts] = useState<CatalystScanSnapshot | null>(null);
  const [catalystError, setCatalystError] = useState<string | null>(null);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [alertsError, setAlertsError] = useState<string | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringStatus | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationRecord[]>([]);
  const [liveState, setLiveState] = useState<"connecting" | "live" | "degraded" | "disabled">(
    "connecting",
  );
  const [liveMarketTick, setLiveMarketTick] = useState<LiveMarketTick | null>(null);
  const portfolioRequest = useRef(0);
  const rollsRequest = useRef(0);
  const detailRequest = useRef(0);
  const catalystRequest = useRef(0);
  const alertsRequest = useRef(0);

  const selectedAccountId =
    portfolio.kind === "ready" ? portfolio.snapshot.selected_account_id : "all";

  const loadPortfolio = useCallback(
    async (accountId = "all", refresh = false, background = false) => {
      const request = ++portfolioRequest.current;
      if (!background) {
        setPortfolio({ kind: "loading" });
        setRisk(null);
      }
      try {
        const snapshot = await fetchPortfolio(accountId, refresh);
        if (request !== portfolioRequest.current) return;
        if (accountId === "all") setAccountOptions(snapshot.accounts);
        setPortfolio({ kind: "ready", snapshot });
        try {
          const riskSnapshot = await fetchPortfolioRisk(accountId);
          if (request === portfolioRequest.current) setRisk(riskSnapshot);
        } catch {
          if (request === portfolioRequest.current) setRisk(null);
        }
      } catch (error: unknown) {
        if (request === portfolioRequest.current) setRisk(null);
        if (request === portfolioRequest.current && !background) {
          setPortfolio({
            kind: "error",
            message: error instanceof Error ? error.message : "Portfolio data is unavailable.",
          });
        }
      }
    },
    [],
  );

  const loadMarkets = useCallback(async () => {
    try {
      const [marketOverview, list] = await Promise.all([fetchMarkets(), fetchWatchlist()]);
      setMarkets(marketOverview);
      setWatchlist(list);
    } catch {
      /* panels remain empty with explicit empty states */
    }
  }, []);

  const loadCatalysts = useCallback(async (accountId = "all") => {
    const request = ++catalystRequest.current;
    // Clear prior account's catalyst state so results never remain "fresh" across accounts.
    setCatalysts(null);
    setCatalystError(null);
    try {
      const scan = await fetchCatalysts(accountId);
      if (request !== catalystRequest.current) return;
      setCatalysts(scan);
      setCatalystError(null);
    } catch (error: unknown) {
      if (request !== catalystRequest.current) return;
      setCatalysts(null);
      setCatalystError(
        error instanceof Error ? error.message : "Catalyst coverage is unavailable.",
      );
    }
  }, []);

  const loadAlerts = useCallback(async (accountId = "all") => {
    const request = ++alertsRequest.current;
    setAlertsError(null);
    try {
      const [rows, recs] = await Promise.all([
        fetchAlerts(accountId),
        fetchRecommendations(accountId),
      ]);
      if (request !== alertsRequest.current) return;
      setAlerts(rows);
      setRecommendations(recs);
    } catch (error: unknown) {
      if (request !== alertsRequest.current) return;
      setAlerts([]);
      setAlertsError(error instanceof Error ? error.message : "Alert center is unavailable.");
    }
  }, []);

  const loadMonitoring = useCallback(async () => {
    try {
      setMonitoring(await fetchMonitoring());
    } catch {
      /* bootstrap monitoring strip remains the fallback */
    }
  }, []);

  const loadRolls = useCallback(async (accountId: string) => {
    const request = ++rollsRequest.current;
    if (accountId === "all") {
      setRolls([]);
      setPatterns(null);
      setHeatmap(null);
      setOrders([]);
      return;
    }
    try {
      setRolls([]);
      setPatterns(null);
      setOrders([]);
      setHeatmap(null);
      const chainRows = await fetchRolls(accountId);
      if (request !== rollsRequest.current) return;
      setRolls(chainRows);
      const [patternResult, orderResult] = await Promise.allSettled([
        fetchRollPatterns(accountId),
        fetchOrders(accountId),
      ]);
      if (request !== rollsRequest.current) return;
      setPatterns(patternResult.status === "fulfilled" ? patternResult.value : null);
      setOrders(orderResult.status === "fulfilled" ? orderResult.value : []);
      const symbol = chainRows[0]?.underlying;
      if (symbol) {
        try {
          const nextHeatmap = await fetchRollHeatmap(accountId, symbol);
          if (request === rollsRequest.current) setHeatmap(nextHeatmap);
        } catch {
          if (request === rollsRequest.current) setHeatmap(null);
        }
      }
    } catch {
      if (request !== rollsRequest.current) return;
      setRolls([]);
      setPatterns(null);
      setHeatmap(null);
      setOrders([]);
    }
  }, []);

  useEffect(() => {
    getBootstrap()
      .then((payload) => setState({ kind: "ready", payload }))
      .catch((error: unknown) =>
        setState({
          kind: "error",
          message: error instanceof Error ? error.message : "The dashboard could not start.",
        }),
      );
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (state.kind !== "ready") return;
    const primary = state.payload.primary_account_id || "all";
    void loadPortfolio().then(() => loadPortfolio(primary, true, true));
    void loadMarkets();
    void loadMonitoring();
    setMonitoring(state.payload.monitoring);
  }, [
    loadMarkets,
    loadMonitoring,
    loadPortfolio,
    state.kind,
    state.kind === "ready" ? state.payload.primary_account_id : null,
    state.kind === "ready" ? state.payload.monitoring : null,
  ]);

  useEffect(() => {
    if (portfolio.kind !== "ready") return;
    void loadRolls(portfolio.snapshot.selected_account_id);
    void loadCatalysts(portfolio.snapshot.selected_account_id);
    void loadAlerts(portfolio.snapshot.selected_account_id);
  }, [loadRolls, loadCatalysts, loadAlerts, portfolio]);

  useEffect(() => {
    if (state.kind !== "ready") return;
    const events = new EventSource("/api/v1/events");
    const refreshStatus = () => {
      void fetchStreamingState().then(setLiveState);
    };
    events.onopen = refreshStatus;
    events.onerror = () => setLiveState("degraded");
    events.onmessage = (message) => {
      const event = JSON.parse(message.data) as {
        event_type: string;
        payload?: { symbol?: string; values?: Record<string, unknown> };
      };
      if (event.event_type === "portfolio.reconciled") {
        void loadPortfolio(selectedAccountId, false, true);
        void loadCatalysts(selectedAccountId);
      }
      if (event.event_type.startsWith("market.") && event.payload?.symbol) {
        const values = event.payload.values ?? {};
        const rawPrice = values.price ?? values.bidPrice ?? values.askPrice;
        setLiveMarketTick({
          symbol: event.payload.symbol,
          price: typeof rawPrice === "number" ? rawPrice : null,
        });
      }
    };
    const statusTimer = window.setInterval(refreshStatus, 30_000);
    return () => {
      events.close();
      window.clearInterval(statusTimer);
    };
  }, [loadCatalysts, loadPortfolio, selectedAccountId, state.kind]);

  const openStrategy = async (strategyId: string) => {
    const request = ++detailRequest.current;
    setSelectedStrategyId(strategyId);
    setStrategyDetail(null);
    try {
      const detail = await fetchStrategyDetail(strategyId);
      if (request === detailRequest.current) setStrategyDetail(detail);
    } catch {
      if (request === detailRequest.current) setStrategyDetail(null);
    }
  };

  const selectAccount = (accountId: string) => {
    setCatalysts(null);
    setCatalystError(null);
    void savePrimaryAccount(accountId);
    void loadPortfolio(accountId);
  };

  const closeStrategy = useCallback(() => {
    detailRequest.current += 1;
    setSelectedStrategyId(null);
    setStrategyDetail(null);
  }, []);

  if (state.kind !== "ready") {
    return <LaunchState state={state} />;
  }

  const snapshot = portfolio.kind === "ready" ? portfolio.snapshot : null;

  return (
    <div className={`app-shell${isPhone ? " phone-readonly" : ""}`} data-layout={isPhone ? "phone" : "desktop"}>
      <NavigationRail
        items={state.payload.navigation}
        activeSection={activeSection}
        onChange={setActiveSection}
      />
      <div className="workspace">
        <WorkspaceHeader
          payload={state.payload}
          portfolio={portfolio}
          accountOptions={accountOptions}
          liveState={liveState}
          liveMarketTick={liveMarketTick}
          onAccountChange={isPhone ? () => undefined : selectAccount}
          onRefresh={() => void loadPortfolio(selectedAccountId, true)}
          theme={theme}
          onThemeChange={() => setTheme(theme === "dark" ? "light" : "dark")}
          readOnlyPhone={isPhone}
        />
        <main id="main-content" className="workspace-main" tabIndex={-1}>
          <h1 className="sr-only">Position Pilot — {activeSection}</h1>
          {isPhone ? (
            <p className="phone-readonly-notice" role="status" aria-live="polite">
              Phone layout is simplified and read-only. Portfolio, alerts, and catalysts remain
              visible. Editing, destructive actions, and recommendation controls are available on a
              wider screen.
            </p>
          ) : null}
          <FoundationNotice liveState={liveState} />
          {portfolio.kind === "ready" && portfolio.snapshot.notice ? (
            <p className="offline-notice" role="status">
              {portfolio.snapshot.notice}
            </p>
          ) : null}

          {activeSection === "Overview" ? (
            <OverviewSection
              portfolio={portfolio}
              risk={risk}
              markets={markets}
              catalysts={catalysts}
              catalystError={catalystError}
              providers={state.payload.providers}
              monitoring={monitoring ?? state.payload.monitoring}
              recommendations={recommendations}
            />
          ) : null}

          {activeSection === "Positions" ? (
            <PositionsSection
              snapshot={snapshot}
              onSelectStrategy={(id) => void openStrategy(id)}
              orders={orders}
              catalysts={catalysts}
            />
          ) : null}

          {activeSection === "Roll analytics" ? (
            <RollsSection
              accountId={selectedAccountId}
              rolls={rolls}
              patterns={patterns}
              heatmap={heatmap}
            />
          ) : null}

          {activeSection === "Markets" ? (
            <MarketsSection
              markets={markets}
              watchlist={watchlist}
              onSaveWatchlist={async (symbols) => setWatchlist(await saveWatchlist(symbols))}
              readOnlyPhone={isPhone}
            />
          ) : null}

          {activeSection === "Alerts" ? (
            <AlertsSection
              alerts={alerts}
              error={alertsError}
              monitoring={monitoring ?? state.payload.monitoring}
              onRefresh={() => void loadAlerts(selectedAccountId)}
              readOnlyPhone={isPhone}
              onAcknowledge={async (alertId) => {
                await acknowledgeAlert(alertId);
                await loadAlerts(selectedAccountId);
              }}
              onSnooze={async (alertId) => {
                await snoozeAlert(alertId, 60);
                await loadAlerts(selectedAccountId);
              }}
              onResolve={async (alertId) => {
                await resolveAlert(alertId);
                await loadAlerts(selectedAccountId);
              }}
              onMute={async (alert) => {
                await muteAlerts({
                  category: alert.category,
                  alert_type: alert.alert_type,
                  symbol: alert.symbol ?? undefined,
                  strategy_type: alert.strategy_type ?? undefined,
                });
                await loadAlerts(selectedAccountId);
              }}
            />
          ) : null}

          {activeSection === "Settings" ? (
            <SettingsSection
              payload={state.payload}
              accountOptions={accountOptions}
              catalystSettings={catalysts?.settings ?? state.payload.catalysts ?? null}
              monitoring={monitoring ?? state.payload.monitoring}
              recommendationSettings={state.payload.recommendations ?? null}
              readOnlyPhone={isPhone}
              onCatalystSettingsSaved={(next) =>
                setCatalysts((current) =>
                  current ? { ...current, settings: next } : current,
                )
              }
              onMonitoringChanged={(next) => setMonitoring(next)}
              onRecommendationSettingsSaved={(next) =>
                setState((current) =>
                  current.kind === "ready"
                    ? { ...current, payload: { ...current.payload, recommendations: next } }
                    : current,
                )
              }
            />
          ) : null}
        </main>
      </div>

      {selectedStrategyId && strategyDetail ? (
        <StrategyDetailDrawer
          key={strategyDetail.strategy.strategy_id}
          detail={strategyDetail}
          onClose={closeStrategy}
          readOnlyPhone={isPhone}
          onSaved={async () => {
            const request = ++detailRequest.current;
            const detail = await fetchStrategyDetail(selectedStrategyId);
            if (request === detailRequest.current) setStrategyDetail(detail);
          }}
        />
      ) : null}
    </div>
  );
}

function LaunchState({ state }: { state: Exclude<AppState, { kind: "ready" }> }) {
  const isError = state.kind === "error";
  return (
    <main className="launch-state">
      <div className="launch-mark" aria-hidden="true">
        P<span>/</span>P
      </div>
      <p className="eyebrow">Local portfolio workstation</p>
      <h1>{isError ? "Session unavailable" : "Position Pilot"}</h1>
      <p className="launch-message">{state.message}</p>
      {isError ? <Unplug aria-hidden="true" /> : <RefreshCw className="spin" aria-hidden="true" />}
    </main>
  );
}

function NavigationRail({
  items,
  activeSection,
  onChange,
}: {
  items: string[];
  activeSection: string;
  onChange: (section: string) => void;
}) {
  return (
    <aside className="navigation-rail" aria-label="Primary navigation">
      <div className="brand-lockup" aria-label="Position Pilot">
        <span>POSITION</span>
        <strong>/</strong>
        <span>PILOT</span>
      </div>
      <nav>
        {items.map((label) => {
          const Icon = navigationIcons[label] ?? CircleDot;
          return (
            <button
              className={label === activeSection ? "nav-item active" : "nav-item"}
              key={label}
              onClick={() => onChange(label)}
              aria-current={label === activeSection ? "page" : undefined}
            >
              <Icon aria-hidden="true" />
              <span>{label}</span>
            </button>
          );
        })}
      </nav>
      <div className="rail-foot">
        <ShieldCheck aria-hidden="true" />
        <span>Read-only</span>
      </div>
    </aside>
  );
}

function WorkspaceHeader({
  payload,
  portfolio,
  accountOptions,
  liveState,
  liveMarketTick,
  onAccountChange,
  onRefresh,
  theme,
  onThemeChange,
  readOnlyPhone = false,
}: {
  payload: BootstrapPayload;
  portfolio: PortfolioState;
  accountOptions: PortfolioAccount[];
  liveState: "connecting" | "live" | "degraded" | "disabled";
  liveMarketTick: LiveMarketTick | null;
  onAccountChange: (accountId: string) => void;
  onRefresh: () => void;
  theme: "dark" | "light";
  onThemeChange: () => void;
  readOnlyPhone?: boolean;
}) {
  const time = useMemo(
    () =>
      new Intl.DateTimeFormat(undefined, {
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
      }).format(new Date(payload.server_time)),
    [payload.server_time],
  );
  return (
    <header className="workspace-header">
      <div className="account-context">
        <span>Workspace</span>
        <select
          aria-label="Account scope"
          value={portfolio.kind === "ready" ? portfolio.snapshot.selected_account_id : "all"}
          onChange={(event) => onAccountChange(event.target.value)}
          disabled={portfolio.kind !== "ready" || readOnlyPhone}
        >
          <option value="all">All accounts</option>
          {accountOptions.map((account) => (
            <option value={account.account_id} key={account.account_id}>
              {account.label}
            </option>
          ))}
        </select>
      </div>
      <div className="header-actions">
        <span className={`local-clock ${liveState}`}>
          <CircleDot aria-hidden="true" />{" "}
          {liveState === "live"
            ? "Live streams"
            : liveState === "disabled"
              ? "Streaming disabled"
              : liveState}
          {liveMarketTick
            ? ` · ${liveMarketTick.symbol}${liveMarketTick.price === null ? "" : ` ${liveMarketTick.price.toFixed(2)}`}`
            : ""}
          {` · ${time}`}
        </span>
        <button className="icon-action" onClick={onRefresh} aria-label="Refresh portfolio">
          <RefreshCw aria-hidden="true" />
        </button>
        <button
          className="icon-action"
          onClick={onThemeChange}
          aria-label={`Use ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
        </button>
        <button className="command-button" type="button" aria-label="Commands (⌘K)">
          <Command aria-hidden="true" />
          <span>Commands</span>
          <kbd>⌘K</kbd>
        </button>
      </div>
    </header>
  );
}

function FoundationNotice({
  liveState,
}: {
  liveState: "connecting" | "live" | "degraded" | "disabled";
}) {
  const message =
    liveState === "live"
      ? "Market and account events are live; REST snapshots remain the five-minute authority."
      : liveState === "disabled"
        ? "Streaming is disabled; the dashboard is using REST and cached snapshots."
        : liveState === "degraded"
          ? "One or more streams are degraded; REST snapshots remain authoritative."
          : "Streams are connecting; REST snapshots remain authoritative.";
  return (
    <section className="foundation-notice" aria-label="Implementation status">
      <span className="status-pip" aria-hidden="true" />
      <p>
        <strong>Catalyst intelligence.</strong> {message}
      </p>
      <span>Phase 6 / 7</span>
    </section>
  );
}

function OverviewSection({
  portfolio,
  risk,
  markets,
  catalysts,
  catalystError,
  providers,
  monitoring,
  recommendations,
}: {
  portfolio: PortfolioState;
  risk: PortfolioRisk | null;
  markets: MarketOverview | null;
  catalysts: CatalystScanSnapshot | null;
  catalystError: string | null;
  providers: BootstrapPayload["providers"];
  monitoring: MonitoringStatus;
  recommendations: RecommendationRecord[];
}) {
  const snapshot = portfolio.kind === "ready" ? portfolio.snapshot : null;
  const promoted = (catalysts?.results ?? []).filter((row) => row.promoted);
  const quietCount = (catalysts?.results ?? []).filter((row) => row.quiet).length;
  const urgentRecs = recommendations
    .filter((row) => row.action && (row.urgency ?? 0) >= 3)
    .slice(0, 5);
  return (
    <>
      <section className="portfolio-masthead" aria-label="Portfolio decision field">
        <div>
          <p className="eyebrow">Portfolio overview</p>
          <h2>Decision field</h2>
        </div>
        {snapshot ? (
          <div className="portfolio-total">
            <span>Net liquidating value</span>
            <strong className="tabular">{currency(snapshot.totals.net_liquidating_value)}</strong>
          </div>
        ) : null}
        <div className="snapshot-state">
          <Clock3 aria-hidden="true" />
          <div>
            <span>Portfolio snapshot</span>
            <strong>
              {snapshot
                ? `${snapshot.state === "cached" ? "Cached" : "Live"} · ${new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date(snapshot.captured_at))}`
                : portfolio.kind === "error"
                  ? "Unavailable"
                  : "Loading"}
            </strong>
          </div>
        </div>
      </section>

      <div className="overview-grid">
        {/* Priority: urgent signals + catalyst scan status first for quick decisions */}
        <div className="priority-band">
          <section className="panel-section" aria-labelledby="rec-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">01 / Action</p>
                <h2 id="rec-heading">Codex signals</h2>
              </div>
              <span className="section-state">
                {urgentRecs.length ? `${urgentRecs.length} urgent` : "quiet"}
              </span>
            </div>
            <ul className="plain-list">
              {urgentRecs.map((row) => (
                <li key={row.recommendation_id} className="recommendation-card">
                  <strong>
                    {row.symbol ?? "Portfolio"} · {row.action ?? "—"}
                  </strong>
                  <span className="muted">
                    {" "}
                    · urgency {row.urgency ?? "—"} · {row.risk ?? "—"} · {row.provider_status}
                  </span>
                  {row.reasoning ? <p className="microcopy">{row.reasoning}</p> : null}
                  {row.error ? (
                    <p className="microcopy" role="status">
                      {row.error}
                    </p>
                  ) : null}
                </li>
              ))}
              {!urgentRecs.length ? (
                <li className="muted">
                  No stored recommendations yet. Open a strategy and run Evaluate, or enable
                  monitoring consent in Settings.
                </li>
              ) : null}
            </ul>
          </section>

          <section className="panel-section" aria-labelledby="catalyst-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">02 / Catalysts</p>
                <h2 id="catalyst-heading">Held-symbol catalysts</h2>
              </div>
              <CatalystFreshness scan={catalysts} error={catalystError} />
            </div>
            {catalystError ? (
              <p className="muted" role="status">
                {catalystError}
              </p>
            ) : null}
            {catalysts?.coverage === "incomplete" || catalysts?.coverage === "offline" ? (
              <p className="coverage-notice" role="status">
                Coverage {catalysts.coverage}
                {catalysts.coverage_notes.length
                  ? ` — ${catalysts.coverage_notes.slice(0, 2).join("; ")}`
                  : ""}
                . Stale or partial news is never used as sole proof of a new move.
              </p>
            ) : null}
            <div className="metric-rail" aria-label="Catalyst summary">
              <div>
                <span>Promoted</span>
                <strong className="tabular">{promoted.length}</strong>
              </div>
              <div>
                <span>Quiet</span>
                <strong className="tabular">{quietCount}</strong>
              </div>
              <div>
                <span>Scanned</span>
                <strong className="tabular">{catalysts?.results.length ?? "—"}</strong>
              </div>
            </div>
            <ul className="catalyst-list compact">
              {(catalysts?.results ?? []).map((row) => (
                <li
                  key={row.symbol}
                  className={row.quiet ? "catalyst-quiet" : "catalyst-promoted"}
                >
                  <div className="catalyst-row-head">
                    <strong>{row.symbol}</strong>
                    <span className={`pill confidence-${row.confidence}`}>
                      {formatConfidence(row.confidence)}
                    </span>
                    {row.move_percent != null ? (
                      <span className="tabular muted">{signed(row.move_percent, 2)}%</span>
                    ) : null}
                    {row.freshness?.state === "stale" || row.cached ? (
                      <span className="pill">Stale cache</span>
                    ) : null}
                  </div>
                  <p>{row.summary || "No confirmed catalyst found"}</p>
                  <p className="microcopy">
                    {formatAttribution(row.attribution)}
                    {row.option_mechanisms.length
                      ? ` · ${row.option_mechanisms.length} option mechanism(s)`
                      : ""}
                  </p>
                </li>
              ))}
              {!catalysts?.results.length && !catalystError ? (
                <li className="muted">Catalysts load after the portfolio snapshot.</li>
              ) : null}
            </ul>
          </section>
        </div>

        <section className="panel-section risk-field" aria-labelledby="risk-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">03 / Exposure</p>
              <h2 id="risk-heading">Risk field</h2>
            </div>
          </div>
          <div className="metric-rail" aria-label="Portfolio metrics">
            <div>
              <span>Accounts</span>
              <strong className="tabular">{snapshot?.accounts.length ?? "—"}</strong>
            </div>
            <div>
              <span>Strategies</span>
              <strong className="tabular">{snapshot?.strategies.length ?? "—"}</strong>
            </div>
            <div>
              <span>Unrealized P/L</span>
              <strong className="tabular pnl">
                {snapshot ? currency(snapshot.totals.unrealized_pnl) : "—"}
              </strong>
            </div>
            <div>
              <span>Buying power</span>
              <strong className="tabular">
                {snapshot ? currency(snapshot.totals.buying_power) : "—"}
              </strong>
            </div>
            <div>
              <span>Net delta</span>
              <strong className="tabular">{risk ? signed(risk.total_delta, 1) : "—"}</strong>
            </div>
            <div>
              <span>Net theta</span>
              <strong className="tabular">{risk ? signed(risk.total_theta, 0) : "—"}</strong>
            </div>
          </div>
          {risk?.concentration?.length ? (
            <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Portfolio concentration table"
            >
              <table className="data-table dense">
                <caption className="sr-only">Concentration by underlying</caption>
                <thead>
                  <tr>
                    <th scope="col">Underlying</th>
                    <th scope="col">Share</th>
                    <th scope="col">Delta</th>
                    <th scope="col">Strategies</th>
                  </tr>
                </thead>
                <tbody>
                  {risk.concentration.slice(0, 6).map((row) => (
                    <tr key={row.underlying}>
                      <td>{row.underlying}</td>
                      <td className="tabular">{pct(row.share_of_portfolio)}</td>
                      <td className="tabular">{signed(row.net_delta, 1)}</td>
                      <td className="tabular">{row.strategy_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>

        <section className="panel-section" aria-labelledby="stress-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">04 / Stress</p>
              <h2 id="stress-heading">Deterministic stress</h2>
            </div>
          </div>
          <ul className="stress-list compact">
            {(risk?.stress ?? []).map((item) => (
              <li key={item.name}>
                <span>{item.label}</span>
                <strong className="tabular">{currency(item.estimated_pnl_change)}</strong>
              </li>
            ))}
            {!risk?.stress?.length ? (
              <li className="muted">Stress awaits a portfolio snapshot.</li>
            ) : null}
          </ul>
        </section>

        <section className="panel-section" aria-labelledby="balances-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">05 / Balances</p>
              <h2 id="balances-heading">Account summary</h2>
            </div>
          </div>
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Account balances table"
          >
            <table className="data-table dense">
              <caption className="sr-only">Account balances</caption>
              <thead>
                <tr>
                  <th scope="col">Account</th>
                  <th scope="col">NLV</th>
                  <th scope="col">Cash</th>
                  <th scope="col">BP</th>
                  <th scope="col">Day P/L</th>
                  <th scope="col">Maint. excess</th>
                  <th scope="col">Day-trade BP</th>
                </tr>
              </thead>
              <tbody>
                {(snapshot?.accounts ?? []).map((account) => (
                  <tr key={account.account_id}>
                    <td>
                      {account.label}
                      <span className="muted block">{account.account_type}</span>
                    </td>
                    <td className="tabular">{currency(account.net_liquidating_value)}</td>
                    <td className="tabular">{currency(account.cash_balance)}</td>
                    <td className="tabular">{currency(account.buying_power)}</td>
                    <td className="tabular">{currency(account.pnl_today)}</td>
                    <td className="tabular">
                      {account.maintenance_excess == null
                        ? "—"
                        : currency(account.maintenance_excess)}
                    </td>
                    <td className="tabular">
                      {account.day_trading_buying_power == null
                        ? "—"
                        : currency(account.day_trading_buying_power)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel-section" aria-labelledby="iv-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">06 / Markets</p>
              <h2 id="iv-heading">IV environment</h2>
            </div>
          </div>
          <div className="metric-rail">
            {(markets?.quotes ?? []).slice(0, 4).map((quote) => (
              <div key={quote.symbol}>
                <span>{quote.symbol}</span>
                <strong className="tabular">{quote.price.toFixed(2)}</strong>
                <em>{quote.iv_environment.replaceAll("_", " ")}</em>
              </div>
            ))}
            {!markets?.quotes?.length ? (
              <div>
                <span>Quotes</span>
                <strong>—</strong>
              </div>
            ) : null}
          </div>
        </section>

        <ProviderLedger providers={providers} />
        <MonitoringStrip monitoring={monitoring} />
      </div>
    </>
  );
}

function AlertsSection({
  alerts,
  error,
  monitoring,
  onRefresh,
  onAcknowledge,
  onSnooze,
  onResolve,
  onMute,
  readOnlyPhone = false,
}: {
  alerts: AlertRecord[];
  error: string | null;
  monitoring: MonitoringStatus;
  onRefresh: () => void;
  onAcknowledge: (alertId: string) => Promise<void>;
  onSnooze: (alertId: string) => Promise<void>;
  onResolve: (alertId: string) => Promise<void>;
  onMute: (alert: AlertRecord) => Promise<void>;
  readOnlyPhone?: boolean;
}) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  return (
    <section className="panel-section" aria-labelledby="alerts-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Alert center</p>
          <h2 id="alerts-heading">Alerts</h2>
        </div>
        <button type="button" onClick={onRefresh}>
          Refresh
        </button>
      </div>
      <p className="muted">
        Risk, catalyst, recommendation, and provider-health events. macOS notifications default to
        symbol · strategy · alert type only.
      </p>
      <p className="microcopy" role="status">
        Monitoring: {monitoring.enabled ? "enabled" : "disabled"}
        {monitoring.notice ? ` · ${monitoring.notice}` : ""}
      </p>
      {error ? (
        <p role="alert">{error}</p>
      ) : null}
      {actionError ? (
        <p role="alert">{actionError}</p>
      ) : null}
      <ul className="alert-list">
        {alerts.map((alert) => (
          <li key={alert.alert_id} className={`alert-card severity-${alert.severity}`}>
            <div className="catalyst-row-head">
              <strong>
                {alert.symbol ?? "System"} · {alert.alert_type.replaceAll("_", " ")}
              </strong>
              <span className="pill">{alert.severity}</span>
              <span className="pill">{alert.category.replaceAll("_", " ")}</span>
              <span className="muted">{alert.resolution}</span>
            </div>
            <p>{alert.title}</p>
            <p className="microcopy">{alert.summary}</p>
            <p className="microcopy">
              {alert.strategy_type ? `${alert.strategy_type} · ` : ""}
              {alert.source} ·{" "}
              {new Intl.DateTimeFormat(undefined, {
                month: "short",
                day: "numeric",
                hour: "numeric",
                minute: "2-digit",
              }).format(new Date(alert.created_at))}
            </p>
            <div className="inline-actions desktop-only-actions" hidden={readOnlyPhone}>
              <button
                type="button"
                disabled={readOnlyPhone || busyId === alert.alert_id}
                onClick={() => {
                  void (async () => {
                    setBusyId(alert.alert_id);
                    setActionError(null);
                    try {
                      await onAcknowledge(alert.alert_id);
                    } catch (err: unknown) {
                      setActionError(err instanceof Error ? err.message : "Acknowledge failed.");
                    } finally {
                      setBusyId(null);
                    }
                  })();
                }}
              >
                Acknowledge
              </button>
              <button
                type="button"
                disabled={readOnlyPhone || busyId === alert.alert_id}
                onClick={() => {
                  void (async () => {
                    setBusyId(alert.alert_id);
                    setActionError(null);
                    try {
                      await onSnooze(alert.alert_id);
                    } catch (err: unknown) {
                      setActionError(err instanceof Error ? err.message : "Snooze failed.");
                    } finally {
                      setBusyId(null);
                    }
                  })();
                }}
              >
                Snooze 1h
              </button>
              <button
                type="button"
                disabled={busyId === alert.alert_id}
                onClick={() => {
                  void (async () => {
                    setBusyId(alert.alert_id);
                    setActionError(null);
                    try {
                      await onResolve(alert.alert_id);
                    } catch (err: unknown) {
                      setActionError(err instanceof Error ? err.message : "Resolve failed.");
                    } finally {
                      setBusyId(null);
                    }
                  })();
                }}
              >
                Resolve
              </button>
              <button
                type="button"
                disabled={busyId === alert.alert_id}
                onClick={() => {
                  void (async () => {
                    setBusyId(alert.alert_id);
                    setActionError(null);
                    try {
                      await onMute(alert);
                    } catch (err: unknown) {
                      setActionError(err instanceof Error ? err.message : "Mute failed.");
                    } finally {
                      setBusyId(null);
                    }
                  })();
                }}
              >
                Mute rule
              </button>
            </div>
          </li>
        ))}
      </ul>
      {!alerts.length && !error ? (
        <p className="muted">No open alerts. Material recommendation changes will appear here.</p>
      ) : null}
    </section>
  );
}

function PositionsSection({
  snapshot,
  onSelectStrategy,
  orders,
  catalysts,
}: {
  snapshot: PortfolioSnapshot | null;
  onSelectStrategy: (id: string) => void;
  orders: OrderRow[];
  catalysts: CatalystScanSnapshot | null;
}) {
  const strategies = snapshot?.strategies ?? [];
  const [strategyPage, setStrategyPage] = useState(0);
  const [orderPage, setOrderPage] = useState(0);
  const catalystBySymbol = useMemo(() => {
    const map = new Map<string, SymbolCatalystResult>();
    for (const row of catalysts?.results ?? []) map.set(row.symbol, row);
    return map;
  }, [catalysts]);

  useEffect(() => {
    setStrategyPage(0);
  }, [snapshot?.snapshot_id, snapshot?.selected_account_id]);

  useEffect(() => {
    setOrderPage(0);
  }, [orders]);

  const strategyPageCount = Math.max(1, Math.ceil(strategies.length / STRATEGY_PAGE_SIZE));
  const safeStrategyPage = Math.min(strategyPage, strategyPageCount - 1);
  const pagedStrategies = strategies.slice(
    safeStrategyPage * STRATEGY_PAGE_SIZE,
    safeStrategyPage * STRATEGY_PAGE_SIZE + STRATEGY_PAGE_SIZE,
  );
  const orderPageCount = Math.max(1, Math.ceil(orders.length / ORDER_PAGE_SIZE));
  const safeOrderPage = Math.min(orderPage, orderPageCount - 1);
  const pagedOrders = orders.slice(
    safeOrderPage * ORDER_PAGE_SIZE,
    safeOrderPage * ORDER_PAGE_SIZE + ORDER_PAGE_SIZE,
  );

  return (
    <div className="stack-lg">
      <section className="panel-section" aria-labelledby="positions-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Strategies</p>
            <h2 id="positions-heading">Positions</h2>
          </div>
          <span className="section-state">{strategies.length} strategies</span>
        </div>
        <p className="microcopy" role="status" aria-live="polite">
          Showing {pagedStrategies.length} of {strategies.length} (page {safeStrategyPage + 1} of{" "}
          {strategyPageCount}). Tables stay windowed so large portfolios remain keyboard-usable.
        </p>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Portfolio strategies table">
          <table className="data-table dense">
            <caption className="sr-only">Grouped strategies with Greeks and P/L</caption>
            <thead>
              <tr>
                <th scope="col">Underlying</th>
                <th scope="col">Strategy</th>
                <th scope="col">Horizon</th>
                <th scope="col">DTE</th>
                <th scope="col">Strikes</th>
                <th scope="col">Catalyst</th>
                <th scope="col">Δ</th>
                <th scope="col">Θ</th>
                <th scope="col">P/L</th>
              </tr>
            </thead>
            <tbody>
              {pagedStrategies.map((strategy) => {
                const catalyst = catalystBySymbol.get(strategy.underlying);
                return (
                  <tr
                    key={strategy.strategy_id}
                    className={catalyst?.quiet === false ? "row-promoted" : undefined}
                  >
                    <th scope="row">
                      <button
                        type="button"
                        className="linkish"
                        onClick={() => onSelectStrategy(strategy.strategy_id)}
                      >
                        {strategy.underlying}
                      </button>
                    </th>
                    <td>{strategy.strategy_type}</td>
                    <td>
                      <span className={`chip horizon-${strategy.horizon}`}>{strategy.horizon}</span>
                    </td>
                    <td className="tabular">{strategy.days_to_expiration ?? "—"}</td>
                    <td className="tabular">{strategy.strikes || "—"}</td>
                    <td>
                      <span
                        className={`catalyst-inline ${
                          catalyst ? (catalyst.quiet ? "quiet" : "promoted") : "quiet"
                        }`}
                        title={catalyst?.summary ?? "No confirmed catalyst found"}
                      >
                        {catalyst
                          ? formatConfidence(catalyst.confidence)
                          : "No confirmed catalyst found"}
                      </span>
                    </td>
                    <td className="tabular">{signed(strategy.total_delta, 2)}</td>
                    <td className="tabular">{signed(strategy.total_theta, 0)}</td>
                    <td className="tabular">{currency(strategy.unrealized_pnl)}</td>
                  </tr>
                );
              })}
              {!strategies.length ? (
                <tr>
                  <td colSpan={9} className="muted">
                    No strategies in the current account scope.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        {strategies.length > STRATEGY_PAGE_SIZE ? (
          <div className="pager" role="navigation" aria-label="Strategy pages">
            <button
              type="button"
              disabled={safeStrategyPage <= 0}
              aria-label="Previous page"
              onClick={() => setStrategyPage((page) => Math.max(0, page - 1))}
            >
              Previous
            </button>
            <span className="microcopy">
              page {safeStrategyPage + 1} of {strategyPageCount}
            </span>
            <button
              type="button"
              disabled={safeStrategyPage >= strategyPageCount - 1}
              aria-label="Next page"
              onClick={() =>
                setStrategyPage((page) => Math.min(strategyPageCount - 1, page + 1))
              }
            >
              Next
            </button>
          </div>
        ) : null}
      </section>

      <section className="panel-section" aria-labelledby="orders-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Read-only</p>
            <h2 id="orders-heading">Order activity</h2>
          </div>
        </div>
        {snapshot?.selected_account_id === "all" ? (
          <p className="muted">Select a single account to load order history and fill linkage.</p>
        ) : (
          <>
            <div className="table-wrap" tabIndex={0} role="region" aria-label="Order activity table">
              <table className="data-table dense">
                <caption className="sr-only">Recent orders with fill linkage</caption>
                <thead>
                  <tr>
                    <th scope="col">Symbol</th>
                    <th scope="col">Action</th>
                    <th scope="col">Status</th>
                    <th scope="col">Qty</th>
                    <th scope="col">Fill</th>
                    <th scope="col">Fills</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedOrders.map((order) => (
                    <tr key={order.order_id}>
                      <td>{order.underlying_symbol || order.symbol}</td>
                      <td>{order.action}</td>
                      <td>{order.status}</td>
                      <td className="tabular">{order.quantity}</td>
                      <td className="tabular">
                        {order.average_fill_price == null
                          ? "—"
                          : order.average_fill_price.toFixed(2)}
                      </td>
                      <td className="tabular">{order.fills.length}</td>
                    </tr>
                  ))}
                  {!orders.length ? (
                    <tr>
                      <td colSpan={6} className="muted">
                        No recent orders for this account.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
            {orders.length > ORDER_PAGE_SIZE ? (
              <div className="pager" role="navigation" aria-label="Order pages">
                <button
                  type="button"
                  disabled={safeOrderPage <= 0}
                  onClick={() => setOrderPage((page) => Math.max(0, page - 1))}
                >
                  Previous
                </button>
                <span className="microcopy">
                  Page {safeOrderPage + 1} / {orderPageCount}
                </span>
                <button
                  type="button"
                  disabled={safeOrderPage >= orderPageCount - 1}
                  onClick={() => setOrderPage((page) => Math.min(orderPageCount - 1, page + 1))}
                >
                  Next
                </button>
              </div>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}

function RollsSection({
  accountId,
  rolls,
  patterns,
  heatmap,
}: {
  accountId: string;
  rolls: RollChain[];
  patterns: RollPatterns | null;
  heatmap: RollHeatmap | null;
}) {
  if (accountId === "all") {
    return (
      <section className="panel-section">
        <h2>Roll analytics</h2>
        <p className="muted">Select a single account to inspect roll chains, credits, patterns, and heatmaps.</p>
      </section>
    );
  }
  return (
    <div className="stack-lg">
      <section className="panel-section" aria-labelledby="rolls-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">History</p>
            <h2 id="rolls-heading">Roll chains</h2>
          </div>
        </div>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Roll chains table">
          <table className="data-table dense">
            <caption className="sr-only">Roll chains and credits</caption>
            <thead>
              <tr>
                <th scope="col">Underlying</th>
                <th scope="col">Strategy</th>
                <th scope="col">Rolls</th>
                <th scope="col">Chain credit</th>
              </tr>
            </thead>
            <tbody>
              {rolls.map((chain) => (
                <tr key={chain.chain_id}>
                  <td>{chain.underlying}</td>
                  <td>{chain.strategy_type}</td>
                  <td className="tabular">{chain.rolls.length}</td>
                  <td className="tabular">
                    {chain.chain_total_credit == null
                      ? "—"
                      : chain.chain_total_credit.toFixed(2)}
                  </td>
                </tr>
              ))}
              {!rolls.length ? (
                <tr>
                  <td colSpan={4} className="muted">
                    No roll history stored for this account yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <div className="two-col">
        <section className="panel-section" aria-labelledby="patterns-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Patterns</p>
              <h2 id="patterns-heading">Pattern analytics</h2>
            </div>
          </div>
          {patterns ? (
            <dl className="fact-grid">
              <div>
                <dt>Total rolls</dt>
                <dd className="tabular">{patterns.total_rolls}</dd>
              </div>
              <div>
                <dt>Win rate</dt>
                <dd className="tabular">{pct(patterns.win_rate)}</dd>
              </div>
              <div>
                <dt>Avg DTE at roll</dt>
                <dd className="tabular">{patterns.avg_dte_at_roll.toFixed(1)}</dd>
              </div>
              <div>
                <dt>Best DTE window</dt>
                <dd className="tabular">
                  {patterns.best_dte_window[0]}–{patterns.best_dte_window[1]}
                </dd>
              </div>
              <div>
                <dt>Avg roll P/L</dt>
                <dd className="tabular">{currency(patterns.avg_roll_pnl)}</dd>
              </div>
              <div>
                <dt>Total roll P/L</dt>
                <dd className="tabular">{currency(patterns.total_pnl)}</dd>
              </div>
            </dl>
          ) : (
            <p className="muted">Pattern analytics unavailable.</p>
          )}
        </section>

        <section className="panel-section" aria-labelledby="heatmap-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Heatmap</p>
              <h2 id="heatmap-heading">Strike × DTE activity</h2>
            </div>
          </div>
          {heatmap && heatmap.strikes.length ? (
            <div className="heatmap" role="img" aria-label={`Roll heatmap for ${heatmap.underlying}`}>
              <div className="heatmap-head">
                <span>Strike</span>
                {heatmap.buckets.map((bucket) => (
                  <span key={bucket}>{bucket}</span>
                ))}
              </div>
              {heatmap.strikes.map((strike) => (
                <div className="heatmap-row" key={strike}>
                  <span className="tabular">${strike.toFixed(0)}</span>
                  {heatmap.buckets.map((bucket) => {
                    const cell = heatmap.cells.find(
                      (item) => item.strike === strike && item.dte_bucket === bucket,
                    );
                    const count = cell?.count ?? 0;
                    return (
                      <span
                        key={`${strike}-${bucket}`}
                        className={`heat-cell heat-${Math.min(count, 4)}`}
                        title={`${count} rolls`}
                      >
                        {count || "·"}
                      </span>
                    );
                  })}
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No heatmap cells yet for this account.</p>
          )}
        </section>
      </div>
    </div>
  );
}

function MarketsSection({
  markets,
  watchlist,
  onSaveWatchlist,
  readOnlyPhone = false,
}: {
  markets: MarketOverview | null;
  watchlist: WatchlistSnapshot | null;
  onSaveWatchlist: (symbols: string[]) => Promise<void>;
  readOnlyPhone?: boolean;
}) {
  const [draft, setDraft] = useState("");
  return (
    <div className="stack-lg">
      <section className="panel-section" aria-labelledby="market-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Overview</p>
            <h2 id="market-heading">Market overview</h2>
          </div>
        </div>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Market overview table">
          <table className="data-table dense">
            <caption className="sr-only">Broad market quotes and IV</caption>
            <thead>
              <tr>
                <th scope="col">Symbol</th>
                <th scope="col">Price</th>
                <th scope="col">IV rank</th>
                <th scope="col">Environment</th>
                <th scope="col">Spread</th>
              </tr>
            </thead>
            <tbody>
              {(markets?.quotes ?? []).map((quote) => (
                <tr key={quote.symbol}>
                  <td>{quote.symbol}</td>
                  <td className="tabular">{quote.price.toFixed(2)}</td>
                  <td className="tabular">
                    {quote.iv_rank == null ? "—" : quote.iv_rank.toFixed(0)}
                  </td>
                  <td>{quote.iv_environment.replaceAll("_", " ")}</td>
                  <td className="tabular">
                    {quote.spread_percent == null ? "—" : `${quote.spread_percent.toFixed(2)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel-section" aria-labelledby="watchlist-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Watchlist</p>
            <h2 id="watchlist-heading">Quotes</h2>
          </div>
        </div>
        {readOnlyPhone ? null : (
          <form
            className="inline-form desktop-only-actions"
            onSubmit={(event) => {
              event.preventDefault();
              const symbol = draft.trim().toUpperCase();
              if (!symbol) return;
              const next = Array.from(new Set([...(watchlist?.symbols ?? []), symbol]));
              void onSaveWatchlist(next).then(() => setDraft(""));
            }}
          >
            <label>
              <span className="sr-only">Add symbol</span>
              <input
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Add symbol"
                aria-label="Add watchlist symbol"
              />
            </label>
            <button type="submit">Add</button>
          </form>
        )}
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Watchlist quotes table">
          <table className="data-table dense">
            <caption className="sr-only">Watchlist quotes</caption>
            <thead>
              <tr>
                <th scope="col">Symbol</th>
                <th scope="col">Price</th>
                <th scope="col">IV</th>
                <th scope="col">IV rank</th>
                {readOnlyPhone ? null : (
                  <th scope="col">
                    <span className="sr-only">Actions</span>
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {(watchlist?.symbols ?? []).map((symbol) => {
                const quote = watchlist?.quotes.find((item) => item.symbol === symbol);
                return (
                  <tr key={symbol}>
                    <td>{symbol}</td>
                    <td className="tabular">{quote ? quote.price.toFixed(2) : "—"}</td>
                    <td className="tabular">
                      {quote?.iv == null ? "—" : `${(quote.iv * 100).toFixed(1)}%`}
                    </td>
                    <td className="tabular">
                      {quote?.iv_rank == null ? "—" : quote.iv_rank.toFixed(0)}
                    </td>
                    {readOnlyPhone ? null : (
                      <td>
                        <button
                          type="button"
                          className="linkish"
                          onClick={() =>
                            void onSaveWatchlist(
                              (watchlist?.symbols ?? []).filter((item) => item !== symbol),
                            )
                          }
                        >
                          Remove
                        </button>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function SettingsSection({
  payload,
  accountOptions,
  catalystSettings,
  monitoring,
  recommendationSettings,
  onCatalystSettingsSaved,
  onMonitoringChanged,
  onRecommendationSettingsSaved,
  readOnlyPhone = false,
}: {
  payload: BootstrapPayload;
  accountOptions: PortfolioAccount[];
  catalystSettings: CatalystSettings | null;
  monitoring: MonitoringStatus;
  recommendationSettings: RecommendationSettings | null;
  onCatalystSettingsSaved: (settings: CatalystSettings) => void;
  onMonitoringChanged: (status: MonitoringStatus) => void;
  onRecommendationSettingsSaved: (settings: RecommendationSettings) => void;
  readOnlyPhone?: boolean;
}) {
  const [cadence, setCadence] = useState(
    String(catalystSettings?.news_cadence_seconds ?? 300),
  );
  const [stockThreshold, setStockThreshold] = useState(
    String(catalystSettings?.stock_move_threshold_pct ?? 2),
  );
  const [etfThreshold, setEtfThreshold] = useState(
    String(catalystSettings?.etf_move_threshold_pct ?? 1),
  );
  const [benzingaEnabled, setBenzingaEnabled] = useState(
    catalystSettings?.benzinga.enabled ?? true,
  );
  const [publicWebEnabled, setPublicWebEnabled] = useState(
    catalystSettings?.public_web?.enabled ?? false,
  );
  const [publicWebSourcesJson, setPublicWebSourcesJson] = useState(
    JSON.stringify(catalystSettings?.public_web?.sources ?? [], null, 2),
  );
  const [benzingaFullText, setBenzingaFullText] = useState(
    Boolean(catalystSettings?.store_full_text?.consent?.benzinga?.active),
  );
  const [saveState, setSaveState] = useState<string | null>(null);
  const [consentBusy, setConsentBusy] = useState(false);
  const [richPreview, setRichPreview] = useState(
    recommendationSettings?.rich_notification_preview ?? false,
  );

  useEffect(() => {
    if (!catalystSettings) return;
    setCadence(String(catalystSettings.news_cadence_seconds));
    setStockThreshold(String(catalystSettings.stock_move_threshold_pct));
    setEtfThreshold(String(catalystSettings.etf_move_threshold_pct));
    setBenzingaEnabled(catalystSettings.benzinga.enabled);
    setPublicWebEnabled(catalystSettings.public_web?.enabled ?? false);
    setPublicWebSourcesJson(
      JSON.stringify(catalystSettings.public_web?.sources ?? [], null, 2),
    );
    setBenzingaFullText(Boolean(catalystSettings.store_full_text?.consent?.benzinga?.active));
  }, [catalystSettings]);

  useEffect(() => {
    if (!recommendationSettings) return;
    setRichPreview(recommendationSettings.rich_notification_preview);
  }, [recommendationSettings]);

  return (
    <div className="stack-lg">
      <p className="muted" role="note">
        Outputs are decision support only. Position Pilot never places, stages, or cancels orders.
      </p>
      <ProviderLedger providers={payload.providers} />

      <section className="panel-section" aria-labelledby="monitoring-consent-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Monitoring</p>
            <h2 id="monitoring-consent-heading">Trading-day consent</h2>
          </div>
        </div>
        <p className="muted">
          Background evaluation runs 7:30 AM–6:00 PM America/New_York on trading days only after
          you opt in. Position Pilot never places trades. Missed intervals are not replayed.
        </p>
        <dl className="cadence-facts">
          <div>
            <dt>Status</dt>
            <dd>{monitoring.enabled ? "Enabled" : "Disabled"}</dd>
          </div>
          <div>
            <dt>Window</dt>
            <dd>
              {monitoring.inside_window ? "Inside" : "Outside"} ·{" "}
              {monitoring.is_holiday
                ? "Holiday"
                : monitoring.is_early_close
                  ? "Early close"
                  : monitoring.is_trading_day
                    ? "Trading day"
                    : "Closed"}
            </dd>
          </div>
          <div>
            <dt>Codex</dt>
            <dd>{monitoring.provider_status ?? payload.providers.codex}</dd>
          </div>
        </dl>
        {monitoring.notice ? <p className="muted" role="status">{monitoring.notice}</p> : null}
        {readOnlyPhone ? null : (
          <div className="inline-actions desktop-only-actions">
            <button
              type="button"
              disabled={consentBusy}
              onClick={() => {
                void (async () => {
                  setConsentBusy(true);
                  try {
                    const result = await saveMonitoringConsent(!monitoring.enabled);
                    onMonitoringChanged(result.status);
                    setSaveState(
                      result.consent.enabled
                        ? "Monitoring consent granted and persisted."
                        : "Monitoring disabled.",
                    );
                  } catch (error: unknown) {
                    setSaveState(
                      error instanceof Error
                        ? error.message
                        : "Could not update monitoring consent.",
                    );
                  } finally {
                    setConsentBusy(false);
                  }
                })();
              }}
            >
              {monitoring.enabled ? "Disable monitoring" : "Enable monitoring"}
            </button>
            <button
              type="button"
              onClick={() => {
                void (async () => {
                  try {
                    await runMonitoringEvaluation(false);
                    onMonitoringChanged(await fetchMonitoring());
                    setSaveState("On-demand evaluation requested.");
                  } catch (error: unknown) {
                    setSaveState(
                      error instanceof Error ? error.message : "Evaluation failed.",
                    );
                  }
                })();
              }}
            >
              Evaluate now
            </button>
          </div>
        )}
      </section>

      <section className="panel-section" aria-labelledby="recommendation-settings-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Recommendations</p>
            <h2 id="recommendation-settings-heading">Provider and notifications</h2>
          </div>
        </div>
        <p className="muted">
          Recommendations use the local Codex CLI with ChatGPT subscription sign-in. API-key
          providers are not available and never selected silently.
        </p>
        <p className="microcopy">
          Provider: {recommendationSettings?.selected_provider ?? "codex-cli"} · API-key fallback:{" "}
          {recommendationSettings?.api_key_fallback_available ? "available" : "unavailable"}
        </p>
        {readOnlyPhone ? (
          <p className="muted">
            Rich notification preview:{" "}
            {recommendationSettings?.rich_notification_preview ? "enabled" : "disabled"}. Editing is
            available on desktop widths.
          </p>
        ) : (
          <form
            className="stack-form desktop-only-actions"
            onSubmit={(event) => {
              event.preventDefault();
              void (async () => {
                try {
                  const next = await saveRecommendationSettings({
                    rich_notification_preview: richPreview,
                  });
                  onRecommendationSettingsSaved(next);
                  setSaveState("Recommendation settings saved.");
                } catch (error: unknown) {
                  setSaveState(
                    error instanceof Error
                      ? error.message
                      : "Could not save recommendation settings.",
                  );
                }
              })();
            }}
          >
            <label className="compact-field checkbox-field">
              <input
                type="checkbox"
                checked={richPreview}
                onChange={(event) => setRichPreview(event.target.checked)}
              />
              Opt in to richer macOS notification previews (still excludes account ids, quantities,
              balances, and P/L)
            </label>
            <button type="submit">Save recommendation settings</button>
          </form>
        )}
      </section>

      <section className="panel-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Accounts</p>
            <h2>Saved identities</h2>
          </div>
        </div>
        <ul className="plain-list">
          {accountOptions.map((account) => (
            <li key={account.account_id}>
              <strong>{account.label}</strong>
              <span className="muted"> · {account.account_type}</span>
            </li>
          ))}
        </ul>
        <p className="muted">
          Brokerage account numbers never leave the local service. Credentials stay in{" "}
          <code>.env</code>.
        </p>
      </section>

      <section className="panel-section" aria-labelledby="catalyst-settings-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Catalysts</p>
            <h2 id="catalyst-settings-heading">News cadence and thresholds</h2>
          </div>
        </div>
        {readOnlyPhone ? (
          <div className="stack-form">
            <p className="microcopy">News cadence: {cadence}s</p>
            <p className="microcopy">Stock move threshold: {stockThreshold}%</p>
            <p className="microcopy">Broad ETF move threshold: {etfThreshold}%</p>
            <p className="microcopy">
              Benzinga: {benzingaEnabled ? "enabled when configured" : "disabled"} · status:{" "}
              {catalystSettings?.benzinga.status ?? "unknown"}
            </p>
            <p className="muted">Catalyst settings are read-only on phone widths.</p>
          </div>
        ) : (
          <form
            className="stack-form desktop-only-actions"
            onSubmit={(event) => {
              event.preventDefault();
              void (async () => {
                setSaveState(null);
                try {
                  let publicSources: Array<{
                    name: string;
                    url_template: string;
                    format?: string;
                    source_tier?: string;
                    enabled?: boolean;
                  }> = [];
                  try {
                    const parsed = JSON.parse(publicWebSourcesJson || "[]") as unknown;
                    if (!Array.isArray(parsed)) {
                      throw new Error("Public web sources must be a JSON array.");
                    }
                    publicSources = parsed as typeof publicSources;
                  } catch (parseError: unknown) {
                    setSaveState(
                      parseError instanceof Error
                        ? parseError.message
                        : "Invalid public web sources JSON.",
                    );
                    return;
                  }
                  const next = await saveCatalystSettings({
                    news_cadence_seconds: Number(cadence),
                    stock_move_threshold_pct: Number(stockThreshold),
                    etf_move_threshold_pct: Number(etfThreshold),
                    benzinga_enabled: benzingaEnabled,
                    public_web_enabled: publicWebEnabled,
                    public_web_sources: publicSources,
                    store_full_text_providers: benzingaFullText ? ["benzinga"] : [],
                    full_text_consent: {
                      benzinga: {
                        enabled: benzingaFullText,
                        agreement_permits_retention_and_ai: benzingaFullText,
                      },
                    },
                    ...(benzingaFullText
                      ? {}
                      : { clear_full_text_provider: "benzinga" }),
                  });
                  onCatalystSettingsSaved(next);
                  setSaveState("Saved locally.");
                } catch (error: unknown) {
                  setSaveState(
                    error instanceof Error ? error.message : "Could not save catalyst settings.",
                  );
                }
              })();
            }}
          >
            <label className="compact-field">
              News cadence (seconds)
              <input
                type="number"
                min={60}
                max={3600}
                value={cadence}
                onChange={(event) => setCadence(event.target.value)}
              />
            </label>
            <label className="compact-field">
              Stock move threshold (%)
              <input
                type="number"
                min={0.1}
                step={0.1}
                value={stockThreshold}
                onChange={(event) => setStockThreshold(event.target.value)}
              />
            </label>
            <label className="compact-field">
              Broad ETF move threshold (%)
              <input
                type="number"
                min={0.1}
                step={0.1}
                value={etfThreshold}
                onChange={(event) => setEtfThreshold(event.target.value)}
              />
            </label>
            <label className="compact-field checkbox-field">
              <input
                type="checkbox"
                checked={benzingaEnabled}
                onChange={(event) => setBenzingaEnabled(event.target.checked)}
              />
              Enable Benzinga premium catalysts when configured
            </label>
            <p className="microcopy">
              Benzinga status: {catalystSettings?.benzinga.status ?? "unknown"}. Provider keys never
              appear in the browser.
            </p>
            <label className="compact-field checkbox-field">
              <input
                type="checkbox"
                checked={publicWebEnabled}
                onChange={(event) => setPublicWebEnabled(event.target.checked)}
              />
              Enable public web/news supplementation (after Massive/Benzinga gaps)
            </label>
            <label className="compact-field">
              Public web sources (JSON array; empty = off)
              <textarea
                rows={4}
                value={publicWebSourcesJson}
                onChange={(event) => setPublicWebSourcesJson(event.target.value)}
                spellCheck={false}
              />
            </label>
            <p className="microcopy">
              {catalystSettings?.public_web?.note ??
                "Public sources respect robots.txt and never bypass paywalls or login walls."}{" "}
              Status: {catalystSettings?.public_web?.status ?? "disabled"}.
            </p>
            <fieldset className="stack-form">
              <legend>Full article text retention (opt-in)</legend>
              <p className="microcopy" role="note">
                {catalystSettings?.full_text_warning ??
                  "Full article text is not stored by default. Only metadata and short excerpts are kept unless you enable a provider and confirm your agreement permits retention and AI processing."}
              </p>
              <label className="compact-field checkbox-field">
                <input
                  type="checkbox"
                  checked={benzingaFullText}
                  onChange={(event) => setBenzingaFullText(event.target.checked)}
                />
                Store Benzinga full text (I confirm my provider agreement permits retention and AI
                processing)
              </label>
              <p className="microcopy">
                Mode: {catalystSettings?.store_full_text?.mode ?? "metadata_and_excerpt_only"}.
                Default providers: none.
              </p>
            </fieldset>
            <button type="submit" className="primary-action">
              Save catalyst settings
            </button>
            {saveState ? <p role="status">{saveState}</p> : null}
          </form>
        )}
      </section>

      <OperationsPanel accountId={payload.primary_account_id || "all"} readOnlyPhone={readOnlyPhone} />
    </div>
  );
}

function StrategyDetailDrawer({
  detail,
  onClose,
  onSaved,
  readOnlyPhone = false,
}: {
  detail: StrategyDetail;
  onClose: () => void;
  onSaved: () => Promise<void>;
  readOnlyPhone?: boolean;
}) {
  const strategy = detail.strategy;
  const panelRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const [thesisDraft, setThesisDraft] = useState({
    purpose: detail.thesis?.purpose ?? "",
    expected_duration: detail.thesis?.expected_duration ?? "",
    target_range: detail.thesis?.target_range ?? "",
    invalidation: detail.thesis?.invalidation ?? "",
    income_or_hedge_intent: detail.thesis?.income_or_hedge_intent ?? "",
    events_to_watch: detail.thesis?.events_to_watch ?? [],
  });
  const [planDraft, setPlanDraft] = useState({
    entry_thesis: detail.trade_plan?.entry_thesis ?? "",
    intended_duration: detail.trade_plan?.intended_duration ?? "",
    profit_target: detail.trade_plan?.profit_target ?? "",
    max_loss: detail.trade_plan?.max_loss ?? "",
    roll_criteria: detail.trade_plan?.roll_criteria ?? "",
    event_exposure: detail.trade_plan?.event_exposure ?? "",
    exit_deadline: detail.trade_plan?.exit_deadline ?? "",
  });
  const [saveError, setSaveError] = useState<string | null>(null);
  const [recommendation, setRecommendation] = useState(detail.recommendation ?? null);
  const [recBusy, setRecBusy] = useState(false);
  const [recMessage, setRecMessage] = useState<string | null>(null);
  const [decisionNote, setDecisionNote] = useState("");
  const prices = detail.chart.bars.map((bar) => bar.close);
  const min = prices.length ? Math.min(...prices) : 0;
  const max = prices.length ? Math.max(...prices) : 1;
  const path = prices
    .map((price, index) => {
      const x = prices.length <= 1 ? 0 : (index / (prices.length - 1)) * 100;
      const y = max === min ? 50 : 100 - ((price - min) / (max - min)) * 100;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(
        panelRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus();
    };
  }, [onClose]);

  const saveAndReload = async (operation: Promise<unknown>) => {
    setSaveError(null);
    try {
      await operation;
      await onSaved();
    } catch (error: unknown) {
      setSaveError(error instanceof Error ? error.message : "The change could not be saved.");
    }
  };

  return (
    <div className="drawer-root" role="dialog" aria-modal="true" aria-labelledby="detail-title">
      <button type="button" className="drawer-backdrop" aria-label="Close detail" onClick={onClose} />
      <aside className="drawer-panel" ref={panelRef}>
        <header className="drawer-header">
          <div>
            <p className="eyebrow">Position detail</p>
            <h2 id="detail-title">
              {strategy.underlying} {strategy.strategy_type}
            </h2>
          </div>
          <button
            type="button"
            className="icon-action"
            onClick={onClose}
            aria-label="Close"
            ref={closeRef}
          >
            <X aria-hidden="true" />
          </button>
        </header>

        <div className="drawer-body stack-lg">
          <section>
            <h3>Chart</h3>
            <svg
              className="sparkline"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              role="img"
              aria-label={`Price chart for ${strategy.underlying} with volume`}
            >
              <title>
                {strategy.underlying} price and volume
                {detail.chart.prior_close != null
                  ? `, prior close ${detail.chart.prior_close.toFixed(2)}`
                  : ""}
              </title>
              <path d={path || "M0 50 L100 50"} fill="none" stroke="currentColor" strokeWidth="1.5" />
              {detail.chart.prior_close != null && max !== min ? (
                <line
                  x1="0"
                  x2="100"
                  y1={100 - ((detail.chart.prior_close - min) / (max - min)) * 100}
                  y2={100 - ((detail.chart.prior_close - min) / (max - min)) * 100}
                  stroke="currentColor"
                  strokeOpacity="0.35"
                  strokeDasharray="2 2"
                />
              ) : null}
              {(detail.chart.event_markers ?? []).map((marker) => {
                const times = detail.chart.bars.map((bar) => new Date(bar.timestamp).getTime());
                if (times.length < 2) return null;
                const start = times[0];
                const end = times[times.length - 1];
                const x =
                  end === start
                    ? 50
                    : ((new Date(marker.timestamp).getTime() - start) / (end - start)) * 100;
                if (x < 0 || x > 100) return null;
                return (
                  <circle
                    key={marker.catalyst_id}
                    cx={x}
                    cy="12"
                    r="1.6"
                    fill="currentColor"
                  >
                    <title>{marker.headline}</title>
                  </circle>
                );
              })}
              {(() => {
                const rawSeries = detail.chart.volume_series?.length
                  ? detail.chart.volume_series
                  : detail.chart.bars.map((bar) => bar.volume ?? 0);
                const stride = Math.max(1, Math.ceil(rawSeries.length / 360));
                const sampled = rawSeries.filter(
                  (_, index) => index % stride === 0 || index === rawSeries.length - 1,
                );
                return sampled.map((volume, index, series) => {
                const peak = Math.max(...series.map((value) => Number(value) || 0), 1);
                const height = ((Number(volume) || 0) / peak) * 18;
                const x =
                  series.length <= 1 ? 0 : (index / (series.length - 1)) * 100 - 0.4;
                return (
                  <rect
                    key={`vol-${index}`}
                    x={x}
                    y={100 - height}
                    width={0.8}
                    height={height}
                    fill="currentColor"
                    opacity={0.25}
                  >
                    <title>{`Volume ${Number(volume).toLocaleString()}`}</title>
                  </rect>
                );
                });
              })()}
            </svg>
            {detail.chart.notice ? <p className="muted">{detail.chart.notice}</p> : null}
            <p className="tabular">
              Underlying {detail.risk.underlying_price?.toFixed(2) ?? "—"} · Prior close{" "}
              {detail.chart.prior_close?.toFixed(2) ?? "—"} · DTE{" "}
              {detail.risk.combined.nearest_dte ?? "—"}
            </p>
            <p className="microcopy">
              {detail.chart.include_extended_hours && detail.chart.extended_hours_truthful
                ? "Extended-hours bars included in the prior-close-to-now window."
                : "Regular-session window (extended hours not present in returned bars)."}
            </p>
            <table className="data-table dense volume-table">
              <caption className="sr-only">Intraday volume series</caption>
              <thead>
                <tr>
                  <th scope="col">Bar</th>
                  <th scope="col">Volume</th>
                </tr>
              </thead>
              <tbody>
                {(detail.chart.volume_series?.length
                  ? detail.chart.volume_series
                  : detail.chart.bars.map((bar) => bar.volume ?? 0)
                )
                  .slice(-8)
                  .map((volume, index) => (
                    <tr key={`volume-row-${index}`}>
                      <td className="tabular">{index + 1}</td>
                      <td className="tabular">{Number(volume).toLocaleString()}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <label className="compact-field">
              Decision horizon
              <select
                value={strategy.horizon}
                onChange={(event) =>
                  void saveAndReload(
                    saveStrategyHorizon(strategy.strategy_id, event.target.value),
                  )
                }
              >
                <option value="strategic">Strategic</option>
                <option value="tactical">Tactical</option>
                <option value="unclassified">Unclassified</option>
              </select>
            </label>
          </section>

          <CatalystDetailPanel catalyst={detail.catalyst ?? null} symbol={strategy.underlying} />

          <section>
            <h3>Legs</h3>
            <div className="table-wrap" tabIndex={0} role="region" aria-label="Strategy legs table">
              <table className="data-table dense">
                <thead>
                  <tr>
                    <th scope="col">Symbol</th>
                    <th scope="col">Qty</th>
                    <th scope="col">Mark</th>
                    <th scope="col">Δ</th>
                    <th scope="col">P/L</th>
                  </tr>
                </thead>
                <tbody>
                  {strategy.legs.map((leg) => (
                    <tr key={leg.symbol}>
                      <td>{leg.symbol}</td>
                      <td className="tabular">
                        {leg.quantity_direction === "Short" ? "-" : "+"}
                        {Math.abs(leg.quantity)}
                      </td>
                      <td className="tabular">{leg.mark_price?.toFixed(2) ?? "—"}</td>
                      <td className="tabular">{leg.delta?.toFixed(3) ?? "—"}</td>
                      <td className="tabular">{currency(leg.unrealized_pnl)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h3>Risk</h3>
            <dl className="fact-grid">
              <div>
                <dt>Remaining max profit</dt>
                <dd className="tabular">
                  {detail.risk.max_profit == null ? "—" : currency(detail.risk.max_profit)}
                </dd>
              </div>
              <div>
                <dt>Remaining max loss</dt>
                <dd className="tabular">
                  {detail.risk.max_loss == null ? "—" : currency(detail.risk.max_loss)}
                </dd>
              </div>
              <div>
                <dt>Breakevens</dt>
                <dd className="tabular">
                  {detail.risk.breakevens.length
                    ? detail.risk.breakevens.map((value) => value.toFixed(2)).join(", ")
                    : "—"}
                </dd>
              </div>
              <div>
                <dt>Δ / Θ / ν</dt>
                <dd className="tabular">
                  {signed(detail.risk.combined.delta ?? 0, 2)} /{" "}
                  {signed(detail.risk.combined.theta ?? 0, 0)} /{" "}
                  {signed(detail.risk.combined.vega ?? 0, 1)}
                </dd>
              </div>
            </dl>
            <p className="microcopy">Bounds and breakevens use current marks, not opening cost basis.</p>
            <ul className="stress-list compact">
              {detail.risk.stress.slice(0, 6).map((item) => (
                <li key={item.name}>
                  <span>{item.label}</span>
                  <strong className="tabular">{currency(item.estimated_pnl_change)}</strong>
                </li>
              ))}
            </ul>
          </section>

          <section aria-labelledby="drawer-recommendation-heading">
            <h3 id="drawer-recommendation-heading">Recommendation</h3>
            {recommendation ? (
              <div className="recommendation-card">
                <p>
                  <strong>{recommendation.action ?? "—"}</strong>
                  <span className="muted">
                    {" "}
                    · urgency {recommendation.urgency ?? "—"} · {recommendation.risk ?? "—"}
                  </span>
                </p>
                <p className="microcopy">
                  Provider {recommendation.provider_status}
                  {recommendation.codex_called ? " · Codex called" : " · fingerprint skip"}
                </p>
                {recommendation.reasoning ? <p>{recommendation.reasoning}</p> : null}
                {recommendation.error ? <p role="status">{recommendation.error}</p> : null}
                <p className="microcopy">
                  Evaluated{" "}
                  {new Intl.DateTimeFormat(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "numeric",
                    minute: "2-digit",
                  }).format(new Date(recommendation.last_evaluated_at))}
                  {recommendation.recommendation_updated_at
                    ? ` · updated ${new Intl.DateTimeFormat(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "numeric",
                        minute: "2-digit",
                      }).format(new Date(recommendation.recommendation_updated_at))}`
                    : ""}
                </p>
                {(detail.recommendation_history ?? []).length ? (
                  <ul className="plain-list">
                    {(detail.recommendation_history ?? []).slice(0, 5).map((entry) => (
                      <li key={entry.history_id} className="microcopy">
                        <strong>{entry.kind.replaceAll("_", " ")}</strong> · {entry.summary}
                        {entry.diff && Object.keys(entry.diff).length ? (
                          <ul className="plain-list">
                            {Object.entries(entry.diff)
                              .filter(([key]) => key !== "day")
                              .slice(0, 6)
                              .map(([key, change]) => {
                                const row = change as { from?: unknown; to?: unknown };
                                return (
                                  <li key={key}>
                                    {key}: {formatDiffValue(row.from)} → {formatDiffValue(row.to)}
                                  </li>
                                );
                              })}
                          </ul>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                ) : null}
                {readOnlyPhone ? null : (
                  <>
                    <label className="compact-field">
                      Decision note
                      <input
                        value={decisionNote}
                        onChange={(event) => setDecisionNote(event.target.value)}
                        maxLength={500}
                      />
                    </label>
                    <div className="inline-actions desktop-only-actions">
                      {(
                        [
                          ["accepted", "Accept"],
                          ["dismissed", "Dismiss"],
                          ["deferred", "Defer"],
                          ["handled_in_tastytrade", "Handled in Tastytrade"],
                        ] as const
                      ).map(([value, label]) => (
                        <button
                          key={value}
                          type="button"
                          disabled={recBusy}
                          onClick={() => {
                            void (async () => {
                              setRecBusy(true);
                              setRecMessage(null);
                              try {
                                await recordTraderDecision(
                                  recommendation.recommendation_id,
                                  value,
                                  decisionNote,
                                );
                                setRecMessage(`Recorded: ${label}`);
                                await onSaved();
                              } catch (error: unknown) {
                                setRecMessage(
                                  error instanceof Error
                                    ? error.message
                                    : "Could not record decision.",
                                );
                              } finally {
                                setRecBusy(false);
                              }
                            })();
                          }}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            ) : (
              <p className="muted">No recommendation stored for this strategy yet.</p>
            )}
            {readOnlyPhone ? null : (
              <div className="inline-actions desktop-only-actions">
                <button
                  type="button"
                  disabled={recBusy}
                  onClick={() => {
                    void (async () => {
                      setRecBusy(true);
                      setRecMessage(null);
                      try {
                        const next = await evaluateStrategyRecommendation(
                          strategy.strategy_id,
                          true,
                        );
                        setRecommendation(next);
                        setRecMessage(
                          next.provider_status === "ok" ||
                            next.provider_status === "skipped_unchanged"
                            ? "Evaluation complete."
                            : `Provider status: ${next.provider_status}${next.error ? ` — ${next.error}` : ""}`,
                        );
                        await onSaved();
                      } catch (error: unknown) {
                        setRecMessage(
                          error instanceof Error ? error.message : "Evaluation failed.",
                        );
                      } finally {
                        setRecBusy(false);
                      }
                    })();
                  }}
                >
                  {recBusy ? "Evaluating…" : "Evaluate with Codex"}
                </button>
              </div>
            )}
            {recMessage ? <p role="status">{recMessage}</p> : null}
          </section>

          <section>
            <h3>{strategy.horizon === "strategic" ? "Thesis" : "Trade plan"}</h3>
            {saveError ? <p role="alert">{saveError}</p> : null}
            {readOnlyPhone ? (
              <p className="muted">Editing is available on desktop widths.</p>
            ) : strategy.horizon === "strategic" ? (
              <form
                className="stack-form desktop-only-actions"
                onSubmit={(event) => {
                  event.preventDefault();
                  void saveAndReload(saveThesis(strategy.strategy_id, thesisDraft));
                }}
              >
                <label>
                  Purpose
                  <textarea
                    value={thesisDraft.purpose}
                    onChange={(event) =>
                      setThesisDraft({ ...thesisDraft, purpose: event.target.value })
                    }
                    rows={3}
                  />
                </label>
                <label>
                  Expected duration
                  <input
                    value={thesisDraft.expected_duration}
                    onChange={(event) =>
                      setThesisDraft({ ...thesisDraft, expected_duration: event.target.value })
                    }
                  />
                </label>
                <label>
                  Target range
                  <input
                    value={thesisDraft.target_range}
                    onChange={(event) =>
                      setThesisDraft({ ...thesisDraft, target_range: event.target.value })
                    }
                  />
                </label>
                <label>
                  Invalidation
                  <textarea
                    value={thesisDraft.invalidation}
                    onChange={(event) =>
                      setThesisDraft({ ...thesisDraft, invalidation: event.target.value })
                    }
                    rows={2}
                  />
                </label>
                <label>
                  Income or hedge intent
                  <textarea
                    value={thesisDraft.income_or_hedge_intent}
                    onChange={(event) =>
                      setThesisDraft({ ...thesisDraft, income_or_hedge_intent: event.target.value })
                    }
                    rows={2}
                  />
                </label>
                <label>
                  Events to watch (comma separated)
                  <input
                    value={thesisDraft.events_to_watch.join(", ")}
                    onChange={(event) =>
                      setThesisDraft({
                        ...thesisDraft,
                        events_to_watch: event.target.value
                          .split(",")
                          .map((item) => item.trim())
                          .filter(Boolean),
                      })
                    }
                  />
                </label>
                <button type="submit">Save thesis</button>
              </form>
            ) : (
              <form
                className="stack-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  void saveAndReload(saveTradePlan(strategy.strategy_id, planDraft));
                }}
              >
                <label>
                  Entry thesis
                  <textarea
                    value={planDraft.entry_thesis}
                    onChange={(event) =>
                      setPlanDraft({ ...planDraft, entry_thesis: event.target.value })
                    }
                    rows={3}
                  />
                </label>
                <label>
                  Intended duration
                  <input
                    value={planDraft.intended_duration}
                    onChange={(event) =>
                      setPlanDraft({ ...planDraft, intended_duration: event.target.value })
                    }
                  />
                </label>
                <label>
                  Profit target
                  <input
                    value={planDraft.profit_target}
                    onChange={(event) =>
                      setPlanDraft({ ...planDraft, profit_target: event.target.value })
                    }
                  />
                </label>
                <label>
                  Maximum loss
                  <input
                    value={planDraft.max_loss}
                    onChange={(event) =>
                      setPlanDraft({ ...planDraft, max_loss: event.target.value })
                    }
                  />
                </label>
                <label>
                  Roll criteria
                  <textarea
                    value={planDraft.roll_criteria}
                    onChange={(event) =>
                      setPlanDraft({ ...planDraft, roll_criteria: event.target.value })
                    }
                    rows={2}
                  />
                </label>
                <label>
                  Event exposure
                  <textarea
                    value={planDraft.event_exposure}
                    onChange={(event) =>
                      setPlanDraft({ ...planDraft, event_exposure: event.target.value })
                    }
                    rows={2}
                  />
                </label>
                <label>
                  Exit deadline
                  <input
                    value={planDraft.exit_deadline}
                    onChange={(event) =>
                      setPlanDraft({ ...planDraft, exit_deadline: event.target.value })
                    }
                  />
                </label>
                <button type="submit">Save trade plan</button>
              </form>
            )}
          </section>

          <section>
            <h3>Events & audit</h3>
            <ul className="timeline">
              {detail.events
                .slice()
                .sort((a, b) => b.timestamp.localeCompare(a.timestamp))
                .slice(0, 12)
                .map((event, index) => (
                  <li key={`${event.timestamp}-${index}`}>
                    <time dateTime={event.timestamp}>
                      {new Intl.DateTimeFormat(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "numeric",
                        minute: "2-digit",
                      }).format(new Date(event.timestamp))}
                    </time>
                    <span>{event.summary}</span>
                  </li>
                ))}
              {!detail.events.length ? <li className="muted">No events recorded yet.</li> : null}
            </ul>
          </section>

          <section>
            <h3>Roll chain</h3>
            <ul className="plain-list">
              {detail.rolls.map((chain) => (
                <li key={chain.chain_id}>
                  {chain.underlying} · {chain.rolls.length} rolls · credit{" "}
                  {chain.chain_total_credit?.toFixed(2) ?? "—"}
                </li>
              ))}
              {!detail.rolls.length ? <li className="muted">No linked rolls.</li> : null}
            </ul>
          </section>
        </div>
      </aside>
    </div>
  );
}

function ProviderLedger({ providers }: { providers: BootstrapPayload["providers"] }) {
  const entries = [
    ["Tastytrade", providers.tastytrade, "Portfolio authority"],
    ["Codex", providers.codex, "Recommendation engine"],
    ["Massive", providers.massive, "Market fallback"],
    ["Benzinga", providers.benzinga, "Premium catalysts"],
  ] as const;
  return (
    <section className="provider-ledger panel-section" aria-labelledby="provider-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">07 / Sources</p>
          <h2 id="provider-heading">Provider ledger</h2>
        </div>
        <ShieldCheck aria-hidden="true" />
      </div>
      <div className="provider-list">
        {entries.map(([name, state, purpose]) => (
          <div className="provider-row" key={name}>
            <span className={`provider-pip ${state}`} aria-hidden="true" />
            <div>
              <strong>{name}</strong>
              <span>{purpose}</span>
            </div>
            <em>
              {state === "configured"
                ? "Configured"
                : state === "not_checked"
                  ? "Check pending"
                  : state === "signed_out"
                    ? "Signed out"
                    : state === "unavailable"
                      ? "Unavailable"
                      : "Not configured"}
            </em>
          </div>
        ))}
      </div>
    </section>
  );
}

function MonitoringStrip({ monitoring }: { monitoring: MonitoringStatus }) {
  return (
    <section className="monitoring-strip panel-section" aria-labelledby="monitoring-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">08 / Cadence</p>
          <h2 id="monitoring-heading">Decision clock</h2>
        </div>
        <Clock3 aria-hidden="true" />
      </div>
      <div className="time-window">
        <div className="time-label start">
          <strong>{monitoring.window_start}</strong>
          <span>Monitor begins</span>
        </div>
        <div className="time-track" aria-hidden="true">
          <span className={`time-fill${monitoring.inside_window ? " active" : ""}`} />
        </div>
        <div className="time-label end">
          <strong>{monitoring.window_end}</strong>
          <span>{monitoring.is_early_close ? "Early close" : "Final check"}</span>
        </div>
      </div>
      <dl className="cadence-facts">
        <div>
          <dt>Risk pulse</dt>
          <dd>{monitoring.risk_refresh_seconds} seconds</dd>
        </div>
        <div>
          <dt>Evaluation</dt>
          <dd>{monitoring.evaluation_minutes} minutes</dd>
        </div>
        <div>
          <dt>Consent</dt>
          <dd>{monitoring.enabled ? "Enabled" : "Disabled"}</dd>
        </div>
        <div>
          <dt>Market clock</dt>
          <dd>New York</dd>
        </div>
      </dl>
      {monitoring.notice ? <p className="microcopy">{monitoring.notice}</p> : null}
    </section>
  );
}

function formatDiffValue(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "string") return value.length > 80 ? `${value.slice(0, 77)}…` : value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    const text = JSON.stringify(value);
    return text.length > 80 ? `${text.slice(0, 77)}…` : text;
  } catch {
    return String(value);
  }
}

function formatConfidence(value: string) {
  if (value === "no_confirmed_catalyst_found") return "No confirmed catalyst found";
  if (value === "confirmed") return "Confirmed";
  if (value === "likely") return "Likely";
  return value.replaceAll("_", " ");
}

function formatAttribution(value: string) {
  if (value === "options_market") return "Options-market driver";
  if (value === "none") return "No confirmed catalyst";
  if (value === "peer") return "Industry / peer-driven";
  if (value === "macro") return "Macro / market-wide";
  if (value === "company") return "Company-specific";
  return value.replaceAll("_", " ");
}

function CatalystFreshness({
  scan,
  error,
}: {
  scan: CatalystScanSnapshot | null;
  error: string | null;
}) {
  if (error) return <span className="section-state">Offline / incomplete</span>;
  if (!scan) return <span className="section-state">Loading</span>;
  const asOf = new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(scan.freshness.as_of));
  return (
    <span className="section-state">
      {scan.coverage} · {asOf}
    </span>
  );
}

function CatalystDetailPanel({
  catalyst,
  symbol,
}: {
  catalyst: SymbolCatalystResult | null;
  symbol: string;
}) {
  const [note, setNote] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  if (!catalyst) {
    return (
      <section>
        <h3>Catalysts</h3>
        <p className="muted">No confirmed catalyst found</p>
      </section>
    );
  }

  const sendFeedback = async (
    kind: "relevant" | "not_related" | "missing_catalyst",
    catalystId?: string,
  ) => {
    setStatus(null);
    try {
      await submitCatalystFeedback({
        kind,
        catalyst_id: catalystId,
        symbol,
        note,
      });
      setStatus("Feedback recorded locally.");
      setNote("");
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : "Feedback failed.");
    }
  };

  return (
    <section>
      <h3>Catalysts</h3>
      <div className="catalyst-row-head">
        <span className={`pill confidence-${catalyst.confidence}`}>
          {formatConfidence(catalyst.confidence)}
        </span>
        <span className="muted">{formatAttribution(catalyst.attribution)}</span>
        {catalyst.move_percent != null ? (
          <span className="tabular">{signed(catalyst.move_percent, 2)}%</span>
        ) : null}
      </div>
      <p>{catalyst.summary}</p>
      {catalyst.coverage !== "complete" ? (
        <p className="coverage-notice" role="status">
          Coverage {catalyst.coverage}
          {catalyst.coverage_notes.length ? `: ${catalyst.coverage_notes.join("; ")}` : ""}
        </p>
      ) : null}
      <ul className="catalyst-list compact">
        {catalyst.catalysts.map((event) => (
          <li key={event.catalyst_id}>
            <strong>{event.headline}</strong>
            <p className="microcopy">
              {event.sources.map((source) => source.name).join(" · ") || "Source attributed"} ·{" "}
              {new Intl.DateTimeFormat(undefined, {
                month: "short",
                day: "numeric",
                hour: "numeric",
                minute: "2-digit",
              }).format(new Date(event.event_at))}
            </p>
            <p className="microcopy">{event.summary}</p>
            <div className="feedback-row">
              <button type="button" onClick={() => void sendFeedback("relevant", event.catalyst_id)}>
                Relevant
              </button>
              <button
                type="button"
                onClick={() => void sendFeedback("not_related", event.catalyst_id)}
              >
                Not related
              </button>
            </div>
          </li>
        ))}
      </ul>
      {catalyst.option_mechanisms.length ? (
        <div className="mechanism-block">
          <h4>Option-market mechanisms</h4>
          <p className="microcopy">Labeled mechanisms — never presented as news.</p>
          <ul className="plain-list">
            {catalyst.option_mechanisms.map((item) => (
              <li key={`${item.kind}-${item.label}`}>
                <strong>{item.label}</strong>
                <span className="muted"> — {item.summary}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {catalyst.option_mechanism_coverage?.length ? (
        <div className="mechanism-block">
          <h4>Mechanism coverage</h4>
          <ul className="plain-list">
            {catalyst.option_mechanism_coverage.map((item) => (
              <li key={`${item.kind}-coverage`}>
                <strong>{item.label}</strong>
                <span className="muted">
                  {" "}
                  — {item.availability.replaceAll("_", " ")}: {item.detail}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {catalyst.social_side_notes.length ? (
        <div className="social-side-note">
          <h4>Social side note</h4>
          <p className="microcopy">Not catalyst evidence.</p>
          <ul className="plain-list">
            {catalyst.social_side_notes.map((item) => (
              <li key={item.note_id}>
                {item.headline} <span className="muted">({item.source_name})</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <label className="compact-field">
        Feedback note
        <input value={note} onChange={(event) => setNote(event.target.value)} />
      </label>
      <button type="button" onClick={() => void sendFeedback("missing_catalyst")}>
        Missing catalyst
      </button>
      {status ? <p role="status">{status}</p> : null}
    </section>
  );
}

export default App;
