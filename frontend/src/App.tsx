import {
  Activity,
  Bell,
  BookOpenText,
  ChartNoAxesCombined,
  ChevronRight,
  CircleDot,
  Clock3,
  Command,
  Gauge,
  Layers3,
  Moon,
  Newspaper,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sun,
  Unplug,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

type ProviderState = "configured" | "not_configured" | "not_checked";

type BootstrapPayload = {
  application: {
    name: string;
    version: string;
    phase: string;
  };
  providers: Record<"tastytrade" | "codex" | "massive" | "benzinga", ProviderState>;
  monitoring: {
    market_timezone: string;
    window_start: string;
    window_end: string;
    evaluation_minutes: number;
    risk_refresh_seconds: number;
  };
  navigation: string[];
  primary_account_id: string;
  data_state: string;
  server_time: string;
};

type PortfolioAccount = {
  account_id: string;
  label: string;
  account_type: string;
  net_liquidating_value: number;
  cash_balance: number;
  buying_power: number;
  positions: Array<{ unrealized_pnl: number }>;
};

type PortfolioSnapshot = {
  snapshot_id: string;
  captured_at: string;
  state: "live" | "cached";
  accounts: PortfolioAccount[];
  strategies: Array<{ strategy_id: string }>;
  totals: {
    net_liquidating_value: number;
    cash_balance: number;
    buying_power: number;
    unrealized_pnl: number;
  };
  selected_account_id: string;
  notice: string | null;
};

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

function providerLabel(state: ProviderState) {
  if (state === "configured") return "Configured";
  if (state === "not_checked") return "Check pending";
  return "Not configured";
}

async function establishSession(): Promise<BootstrapPayload> {
  const url = new URL(window.location.href);
  const launchToken = url.searchParams.get("launch_token");

  if (launchToken) {
    const exchange = await fetch("/api/v1/session/exchange", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ launch_token: launchToken }),
    });
    if (!exchange.ok) throw new Error("The secure launch link is invalid or expired.");
    url.searchParams.delete("launch_token");
    window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  }

  const response = await fetch("/api/v1/bootstrap", {
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  if (response.status === 401) {
    throw new Error("This dashboard session expired. Run `pilot dashboard` to open a new one.");
  }
  if (!response.ok) throw new Error("Position Pilot could not initialize the local dashboard.");
  return response.json() as Promise<BootstrapPayload>;
}

let bootstrapPromise: Promise<BootstrapPayload> | undefined;

function getBootstrap() {
  bootstrapPromise ??= establishSession();
  return bootstrapPromise;
}

async function fetchPortfolio(accountId = "all", refresh = false): Promise<PortfolioSnapshot> {
  const search = new URLSearchParams({ account_id: accountId });
  if (refresh) search.set("refresh", "true");
  const response = await fetch(`/api/v1/portfolio?${search}`, {
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new Error("Portfolio data is currently unavailable.");
  return response.json() as Promise<PortfolioSnapshot>;
}

async function savePrimaryAccount(accountId: string) {
  const response = await fetch("/api/v1/settings/primary-account", {
    method: "PUT",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ account_id: accountId }),
  });
  if (!response.ok) throw new Error("The primary account preference could not be saved.");
}

function currency(value: number) {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

function App() {
  const [state, setState] = useState<AppState>({
    kind: "loading",
    message: "Securing local session…",
  });
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [activeSection, setActiveSection] = useState("Overview");
  const [portfolio, setPortfolio] = useState<PortfolioState>({ kind: "loading" });
  const [accountOptions, setAccountOptions] = useState<PortfolioAccount[]>([]);

  const loadPortfolio = useCallback(
    async (accountId = "all", refresh = false, background = false) => {
      if (!background) setPortfolio({ kind: "loading" });
      try {
        const snapshot = await fetchPortfolio(accountId, refresh);
        if (accountId === "all") setAccountOptions(snapshot.accounts);
        setPortfolio({ kind: "ready", snapshot });
      } catch (error: unknown) {
        if (!background) {
          setPortfolio({
            kind: "error",
            message: error instanceof Error ? error.message : "Portfolio data is unavailable.",
          });
        }
      }
    },
    [],
  );

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
    void loadPortfolio().then(() =>
      loadPortfolio(state.payload.primary_account_id || "all", true, true),
    );
  }, [loadPortfolio, state.kind]);

  const selectAccount = (accountId: string) => {
    void savePrimaryAccount(accountId);
    void loadPortfolio(accountId);
  };

  if (state.kind !== "ready") {
    return <LaunchState state={state} />;
  }

  return (
    <div className="app-shell">
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
          onAccountChange={selectAccount}
          theme={theme}
          onThemeChange={() => setTheme(theme === "dark" ? "light" : "dark")}
        />
        <main id="main-content" className="overview" tabIndex={-1}>
          <FoundationNotice />
          <PortfolioMasthead portfolio={portfolio} />
          {portfolio.kind === "ready" && portfolio.snapshot.notice ? (
            <p className="offline-notice" role="status">{portfolio.snapshot.notice}</p>
          ) : null}
          <div className="overview-grid">
            <RiskField portfolio={portfolio} />
            <CatalystField />
            <ProviderLedger providers={state.payload.providers} />
            <MonitoringStrip monitoring={state.payload.monitoring} />
          </div>
        </main>
      </div>
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
  onAccountChange,
  theme,
  onThemeChange,
}: {
  payload: BootstrapPayload;
  portfolio: PortfolioState;
  accountOptions: PortfolioAccount[];
  onAccountChange: (accountId: string) => void;
  theme: "dark" | "light";
  onThemeChange: () => void;
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
          disabled={portfolio.kind !== "ready"}
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
        <span className="local-clock">
          <CircleDot aria-hidden="true" /> Local service · {time}
        </span>
        <button className="icon-action" onClick={onThemeChange} aria-label={`Use ${theme === "dark" ? "light" : "dark"} theme`}>
          {theme === "dark" ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
        </button>
        <button className="command-button" type="button">
          <Command aria-hidden="true" />
          <span>Commands</span>
          <kbd>⌘K</kbd>
        </button>
      </div>
    </header>
  );
}

function FoundationNotice() {
  return (
    <section className="foundation-notice" aria-label="Implementation status">
      <span className="status-pip" aria-hidden="true" />
      <p>
        <strong>Snapshot service online.</strong> Portfolio state is versioned, account-scoped, and available from durable local cache.
      </p>
      <span>Phase 2 / 7</span>
    </section>
  );
}

function PortfolioMasthead({ portfolio }: { portfolio: PortfolioState }) {
  const snapshot = portfolio.kind === "ready" ? portfolio.snapshot : null;
  return (
    <section className="portfolio-masthead">
      <div>
        <p className="eyebrow">Portfolio overview</p>
        <h1>Decision field</h1>
      </div>
      {snapshot ? (
        <div className="portfolio-total">
          <span>Net liquidating value</span>
          <strong>{currency(snapshot.totals.net_liquidating_value)}</strong>
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
  );
}

function RiskField({ portfolio }: { portfolio: PortfolioState }) {
  const snapshot = portfolio.kind === "ready" ? portfolio.snapshot : null;
  const metrics = snapshot
    ? [
        ["Accounts", String(snapshot.accounts.length)],
        ["Strategies", String(snapshot.strategies.length)],
        ["Unrealized P/L", currency(snapshot.totals.unrealized_pnl)],
        ["Buying power", currency(snapshot.totals.buying_power)],
      ]
    : [
        ["Accounts", "—"],
        ["Strategies", "—"],
        ["Unrealized P/L", "—"],
        ["Buying power", "—"],
      ];
  return (
    <section className="risk-field panel-section" aria-labelledby="risk-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">01 / Exposure</p>
          <h2 id="risk-heading">Risk field</h2>
        </div>
        <span className="section-state">{snapshot ? `${snapshot.accounts.length} account scope` : "Data pending"}</span>
      </div>
      <div className="empty-measure">
        <div className="measure-axis" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>
        <div>
          <Gauge aria-hidden="true" />
          <h3>{snapshot ? "Atomic portfolio snapshot" : "No portfolio snapshot yet"}</h3>
          <p>
            {snapshot
              ? `${snapshot.strategies.length} account-scoped strategies are loaded from one coherent generation.`
              : portfolio.kind === "error"
                ? portfolio.message
                : "Exposure, Greeks, stress scenarios, and strategy urgency will resolve here as one coherent snapshot."}
          </p>
        </div>
      </div>
      <div className="metric-rail" aria-label="Pending portfolio metrics">
        {metrics.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function CatalystField() {
  return (
    <section className="catalyst-field panel-section" aria-labelledby="catalyst-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">02 / Evidence</p>
          <h2 id="catalyst-heading">Catalyst field</h2>
        </div>
        <Newspaper aria-hidden="true" />
      </div>
      <div className="catalyst-empty">
        <span className="event-line" aria-hidden="true" />
        <div>
          <strong>No held symbols loaded</strong>
          <p>Confirmed, likely, and no-catalyst findings will appear with source provenance—never a forced story.</p>
        </div>
      </div>
      <button className="text-action" type="button" disabled>
        Open catalyst timeline <ChevronRight aria-hidden="true" />
      </button>
    </section>
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
          <p className="eyebrow">03 / Sources</p>
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
            <em>{providerLabel(state)}</em>
          </div>
        ))}
      </div>
    </section>
  );
}

function MonitoringStrip({ monitoring }: { monitoring: BootstrapPayload["monitoring"] }) {
  return (
    <section className="monitoring-strip panel-section" aria-labelledby="monitoring-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">04 / Cadence</p>
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
          <span className="time-fill" />
          {Array.from({ length: 8 }, (_, index) => (
            <i key={index} />
          ))}
        </div>
        <div className="time-label end">
          <strong>{monitoring.window_end}</strong>
          <span>Final check</span>
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
          <dt>Market clock</dt>
          <dd>New York</dd>
        </div>
      </dl>
    </section>
  );
}

export default App;
