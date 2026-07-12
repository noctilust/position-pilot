import type {
  AlertRecord,
  BootstrapPayload,
  CatalystScanSnapshot,
  CatalystSettings,
  MarketOverview,
  MarketQuote,
  MonitoringStatus,
  OrderRow,
  PortfolioRisk,
  PortfolioSnapshot,
  RecommendationRecord,
  RecommendationSettings,
  RollChain,
  RollHeatmap,
  RollPatterns,
  StrategyDetail,
  SymbolCatalystResult,
  TraderDecision,
  WatchlistSnapshot,
} from "./types";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    ...init,
  });
  if (response.status === 401) {
    throw new Error("This dashboard session expired. Run `pilot dashboard` to open a new one.");
  }
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${path}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export async function establishSession(): Promise<BootstrapPayload> {
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

  return api<BootstrapPayload>("/api/v1/bootstrap");
}

let bootstrapPromise: Promise<BootstrapPayload> | undefined;

export function getBootstrap() {
  bootstrapPromise ??= establishSession();
  return bootstrapPromise;
}

export function fetchPortfolio(accountId = "all", refresh = false) {
  const search = new URLSearchParams({ account_id: accountId });
  if (refresh) search.set("refresh", "true");
  return api<PortfolioSnapshot>(`/api/v1/portfolio?${search}`);
}

export function fetchPortfolioRisk(accountId = "all") {
  const search = new URLSearchParams({ account_id: accountId });
  return api<PortfolioRisk>(`/api/v1/portfolio/risk?${search}`);
}

export function savePrimaryAccount(accountId: string) {
  return api<void>("/api/v1/settings/primary-account", {
    method: "PUT",
    body: JSON.stringify({ account_id: accountId }),
  });
}

export function fetchStrategyDetail(strategyId: string) {
  return api<StrategyDetail>(`/api/v1/strategies/${strategyId}`);
}

export function saveStrategyHorizon(strategyId: string, horizon: string) {
  return api(`/api/v1/strategies/${strategyId}/horizon`, {
    method: "PATCH",
    body: JSON.stringify({ horizon }),
  });
}

export function saveThesis(strategyId: string, payload: Record<string, unknown>) {
  return api(`/api/v1/strategies/${strategyId}/thesis`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function saveTradePlan(strategyId: string, payload: Record<string, unknown>) {
  return api(`/api/v1/strategies/${strategyId}/trade-plan`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function fetchMarkets() {
  return api<MarketOverview>("/api/v1/markets");
}

export function fetchQuote(symbol: string) {
  return api<MarketQuote>(`/api/v1/markets/${encodeURIComponent(symbol)}`);
}

export function fetchWatchlist() {
  return api<WatchlistSnapshot>("/api/v1/watchlist");
}

export function saveWatchlist(symbols: string[]) {
  return api<WatchlistSnapshot>("/api/v1/watchlist", {
    method: "PUT",
    body: JSON.stringify({ symbols }),
  });
}

export function fetchOrders(accountId: string) {
  return api<OrderRow[]>(`/api/v1/accounts/${accountId}/orders`);
}

export function fetchRolls(accountId: string, symbol?: string) {
  const search = new URLSearchParams();
  if (symbol) search.set("symbol", symbol);
  const query = search.toString();
  return api<RollChain[]>(
    `/api/v1/accounts/${accountId}/rolls${query ? `?${query}` : ""}`,
  );
}

export function fetchRollPatterns(accountId: string, symbol?: string) {
  const search = new URLSearchParams();
  if (symbol) search.set("symbol", symbol);
  const query = search.toString();
  return api<RollPatterns>(
    `/api/v1/accounts/${accountId}/rolls/patterns${query ? `?${query}` : ""}`,
  );
}

export function fetchRollHeatmap(accountId: string, symbol: string) {
  const search = new URLSearchParams({ symbol });
  return api<RollHeatmap>(`/api/v1/accounts/${accountId}/rolls/heatmap?${search}`);
}

export function fetchCatalysts(accountId = "all") {
  const search = new URLSearchParams({ account_id: accountId });
  return api<CatalystScanSnapshot>(`/api/v1/catalysts?${search}`);
}

export function fetchCatalyst(symbol: string) {
  return api<SymbolCatalystResult>(`/api/v1/catalysts/${encodeURIComponent(symbol)}`);
}

export function submitCatalystFeedback(payload: {
  kind: "relevant" | "not_related" | "missing_catalyst";
  catalyst_id?: string | null;
  symbol?: string | null;
  note?: string;
}) {
  return api("/api/v1/catalysts/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchCatalystSettings() {
  return api<CatalystSettings>("/api/v1/settings/catalysts");
}

export function saveCatalystSettings(payload: {
  benzinga_enabled?: boolean;
  news_cadence_seconds?: number;
  stock_move_threshold_pct?: number;
  etf_move_threshold_pct?: number;
}) {
  return api<CatalystSettings>("/api/v1/settings/catalysts", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function fetchRecommendations(accountId = "all") {
  const search = new URLSearchParams({ account_id: accountId });
  return api<RecommendationRecord[]>(`/api/v1/recommendations?${search}`);
}

export function evaluateStrategyRecommendation(strategyId: string, force = false) {
  return api<RecommendationRecord>(`/api/v1/strategies/${strategyId}/recommend`, {
    method: "POST",
    body: JSON.stringify({ force }),
  });
}

export function recordTraderDecision(
  recommendationId: string,
  decision: string,
  note = "",
) {
  return api<TraderDecision>(`/api/v1/recommendations/${recommendationId}/decisions`, {
    method: "POST",
    body: JSON.stringify({ decision, note }),
  });
}

export function fetchAlerts(accountId = "all") {
  const search = new URLSearchParams({ account_id: accountId });
  return api<AlertRecord[]>(`/api/v1/alerts?${search}`);
}

export function acknowledgeAlert(alertId: string) {
  return api<AlertRecord>(`/api/v1/alerts/${alertId}/acknowledge`, { method: "POST" });
}

export function snoozeAlert(alertId: string, minutes = 60) {
  return api<AlertRecord>(`/api/v1/alerts/${alertId}/snooze`, {
    method: "POST",
    body: JSON.stringify({ minutes }),
  });
}

export function resolveAlert(alertId: string) {
  return api<AlertRecord>(`/api/v1/alerts/${alertId}/resolve`, { method: "POST" });
}

export function muteAlerts(payload: {
  category?: string;
  alert_type?: string;
  symbol?: string;
  strategy_type?: string;
}) {
  return api("/api/v1/alerts/mute", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchMonitoring() {
  return api<MonitoringStatus>("/api/v1/monitoring");
}

export function saveMonitoringConsent(enabled: boolean) {
  return api<{ consent: { enabled: boolean }; status: MonitoringStatus }>(
    "/api/v1/monitoring/consent",
    {
      method: "PUT",
      body: JSON.stringify({ enabled }),
    },
  );
}

export function runMonitoringEvaluation(force = false) {
  return api<Record<string, unknown>>("/api/v1/monitoring/evaluate", {
    method: "POST",
    body: JSON.stringify({ force }),
  });
}

export function saveRecommendationSettings(payload: {
  rich_notification_preview?: boolean;
}) {
  return api<RecommendationSettings>("/api/v1/settings/recommendations", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function fetchStreamingState(): Promise<
  "connecting" | "live" | "degraded" | "disabled"
> {
  const response = await fetch("/api/v1/streaming/status", {
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) return "degraded";
  const status = (await response.json()) as Record<string, { state: string }>;
  const states = Object.values(status).map((item) => item.state);
  if (states.length > 0 && states.every((state) => state === "live")) return "live";
  if (states.length > 0 && states.every((state) => state === "disabled")) return "disabled";
  if (states.some((state) => ["degraded", "unavailable", "stale"].includes(state))) {
    return "degraded";
  }
  return "connecting";
}

export function currency(value: number) {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

export function signed(value: number, digits = 2) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}`;
}

export function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}
