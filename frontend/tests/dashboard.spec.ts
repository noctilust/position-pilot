import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

/**
 * Controllable EventSource test double installed before app boot.
 * Keeps the SSE pipe open (no short-body closure) so production onerror
 * semantics stay strict while tests inject Quote / error / reopen.
 */
async function installControllableEventSource(page: Page) {
  await page.addInitScript(() => {
    type Handler = ((ev: Event) => void) | null;
    type MessageHandler = ((ev: MessageEvent) => void) | null;

    class ControllableEventSource {
      static instances: ControllableEventSource[] = [];
      url: string;
      withCredentials = false;
      readyState = 0;
      onopen: Handler = null;
      onerror: Handler = null;
      onmessage: MessageHandler = null;
      readonly CONNECTING = 0;
      readonly OPEN = 1;
      readonly CLOSED = 2;
      private listeners = new Map<string, Set<(ev: Event) => void>>();

      constructor(url: string | URL) {
        this.url = String(url);
        ControllableEventSource.instances.push(this);
        const w = window as unknown as {
          __sseInstances: ControllableEventSource[];
          __sseEmit: (payload: unknown) => void;
          __sseError: () => void;
          __sseReopen: () => void;
        };
        w.__sseInstances = ControllableEventSource.instances;
        w.__sseEmit = (payload: unknown) => {
          const data =
            typeof payload === "string" ? payload : JSON.stringify(payload);
          for (const inst of ControllableEventSource.instances) {
            if (inst.readyState === inst.CLOSED) continue;
            const ev = new MessageEvent("message", { data });
            inst.onmessage?.(ev);
            inst._dispatch("message", ev);
          }
        };
        w.__sseError = () => {
          for (const inst of ControllableEventSource.instances) {
            // Transport failed; leave closed so no accidental further messages.
            inst.readyState = inst.CLOSED;
            const ev = new Event("error");
            inst.onerror?.(ev);
            inst._dispatch("error", ev);
          }
        };
        w.__sseReopen = () => {
          for (const inst of ControllableEventSource.instances) {
            inst.readyState = inst.OPEN;
            const ev = new Event("open");
            inst.onopen?.(ev);
            inst._dispatch("open", ev);
          }
        };
        queueMicrotask(() => {
          this.readyState = this.OPEN;
          const ev = new Event("open");
          this.onopen?.(ev);
          this._dispatch("open", ev);
        });
      }

      addEventListener(type: string, listener: (ev: Event) => void) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type)!.add(listener);
      }

      removeEventListener(type: string, listener: (ev: Event) => void) {
        this.listeners.get(type)?.delete(listener);
      }

      _dispatch(type: string, ev: Event) {
        this.listeners.get(type)?.forEach((fn) => fn(ev));
      }

      close() {
        this.readyState = this.CLOSED;
      }
    }

    Object.defineProperty(ControllableEventSource, "CONNECTING", { value: 0 });
    Object.defineProperty(ControllableEventSource, "OPEN", { value: 1 });
    Object.defineProperty(ControllableEventSource, "CLOSED", { value: 2 });
    (window as unknown as { EventSource: typeof ControllableEventSource }).EventSource =
      ControllableEventSource;
  });
}

async function sseEmit(page: Page, payload: unknown) {
  await page.evaluate((data) => {
    (window as unknown as { __sseEmit: (p: unknown) => void }).__sseEmit(data);
  }, payload);
}

async function sseError(page: Page) {
  await page.evaluate(() => {
    (window as unknown as { __sseError: () => void }).__sseError();
  });
}

async function sseReopen(page: Page) {
  await page.evaluate(() => {
    (window as unknown as { __sseReopen: () => void }).__sseReopen();
  });
}

/** Shared MU strangle fixture for live P/L browser tests. */
function muLivePortfolioBody(snapshotId: string, overrides?: {
  callPnlOpen?: number;
  callMark?: number;
  callRaw?: number;
  strategyPnlOpen?: number;
}) {
  const callMark = overrides?.callMark ?? 4.05;
  const callRaw = overrides?.callRaw ?? 1196;
  const callPnlOpen = overrides?.callPnlOpen ?? 1557;
  const strategyPnlOpen = overrides?.strategyPnlOpen ?? 1345;
  return {
    schema_version: 1,
    snapshot_id: snapshotId,
    captured_at: "2026-07-13T16:30:00Z",
    state: "live",
    freshness: {
      as_of: "2026-07-13T16:30:00Z",
      provider: "tastytrade",
      state: "fresh",
    },
    accounts: [
      {
        account_id: "public-account-id",
        label: "Individual 1",
        account_type: "Individual",
        net_liquidating_value: 25000,
        cash_balance: 5000,
        buying_power: 10000,
        pnl_today: 0,
        positions: [],
      },
    ],
    strategies: [
      {
        strategy_id: "mu-strangle-live",
        account_id: "public-account-id",
        underlying: "MU",
        strategy_type: "Short Strangle",
        expiration_date: "2026-07-31",
        days_to_expiration: 18,
        quantity: 1,
        strikes: "$800/$1400",
        unrealized_pnl: -212 + callRaw,
        unrealized_pnl_percent: 12,
        pnl_open: strategyPnlOpen,
        pnl_open_percent: 18,
        roll_adjustment: 361,
        roll_count: 1,
        total_delta: -0.1,
        total_theta: 20,
        horizon: "tactical",
        legs: [
          {
            symbol: "MU    260731P00800000",
            underlying_symbol: "MU",
            quantity: 1,
            quantity_direction: "Short",
            position_type: "Equity Option",
            strike_price: 800,
            option_type: "P",
            expiration_date: "2026-07-31",
            days_to_expiration: 18,
            mark_price: 4.62,
            market_value: 462,
            cost_basis: 250,
            unrealized_pnl: -212,
            unrealized_pnl_percent: null,
            pnl_open: -212,
            pnl_open_basis: 250,
            roll_adjustment: 0,
            roll_count: 0,
            roll_history_status: "none",
            delta: -0.2,
            gamma: 0.01,
            theta: 10,
            vega: -0.1,
            implied_volatility: 0.4,
            multiplier: 100,
            horizon: "tactical",
          },
          {
            symbol: "MU    260731C01400000",
            underlying_symbol: "MU",
            quantity: 1,
            quantity_direction: "Short",
            position_type: "Equity Option",
            strike_price: 1400,
            option_type: "C",
            expiration_date: "2026-07-31",
            days_to_expiration: 18,
            mark_price: callMark,
            market_value: callMark * 100,
            cost_basis: 1601,
            unrealized_pnl: callRaw,
            unrealized_pnl_percent: null,
            pnl_open: callPnlOpen,
            pnl_open_percent: 25,
            pnl_open_basis: 1962,
            roll_adjustment: 361,
            roll_count: 1,
            roll_history_status: "complete",
            roll_chain_id: "call-chain",
            delta: 0.1,
            gamma: 0.01,
            theta: 10,
            vega: -0.1,
            implied_volatility: 0.4,
            multiplier: 100,
            horizon: "tactical",
          },
        ],
      },
    ],
    totals: {
      net_liquidating_value: 25000,
      cash_balance: 5000,
      buying_power: 10000,
      unrealized_pnl: -212 + callRaw,
    },
    selected_account_id: "public-account-id",
    notice: null,
  };
}

const LIVE_CALL_QUOTE = {
  event_type: "market.Quote",
  payload: {
    symbol: ".MU260731C1400",
    values: {
      eventType: "Quote",
      eventSymbol: ".MU260731C1400",
      bidPrice: 3.7,
      askPrice: 3.75,
      bidSize: 10,
      askSize: 12,
    },
  },
};

async function mockDashboardApis(page: Page) {
  await page.route("**/api/v1/session/exchange", (route) =>
    route.fulfill({ status: 204 }),
  );
  await page.route("**/api/v1/events", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: ": browser smoke heartbeat\n\n",
    }),
  );
  await page.route("**/api/v1/bootstrap", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        application: { name: "Position Pilot", version: "test", phase: "hardening-retirement" },
        providers: {
          tastytrade: "configured",
          codex: "configured",
          massive: "configured",
          benzinga: "not_configured",
        },
        monitoring: {
          market_timezone: "America/New_York",
          window_start: "07:30",
          window_end: "18:00",
          evaluation_minutes: 30,
          risk_refresh_seconds: 60,
          enabled: false,
          consented: false,
          inside_window: true,
          is_trading_day: true,
          is_holiday: false,
          is_early_close: false,
          provider_status: "configured",
          running: false,
          notice: "Monitoring is disabled until you grant onboarding consent.",
          last_evaluation_at: null,
        },
        recommendations: {
          api_key_fallback_enabled: false,
          selected_provider: "codex-cli",
          rich_notification_preview: false,
        },
        catalysts: {
          stock_move_threshold_pct: 2,
          etf_move_threshold_pct: 1,
          news_cadence_seconds: 300,
          benzinga: { enabled: false, status: "disabled" },
          scheduled_window_hours: 72,
        },
        navigation: ["Overview", "Positions", "Roll analytics", "Markets", "Alerts", "Settings"],
        primary_account_id: "public-account-id",
        data_state: "ready",
        server_time: "2026-07-11T16:30:00Z",
      }),
    }),
  );
  await page.route("**/api/v1/settings/primary-account**", (route) =>
    route.fulfill({ status: 204 }),
  );
  await page.route("**/api/v1/alerts**", (route) =>
    route.fulfill({ contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/v1/recommendations**", (route) =>
    route.fulfill({ contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/v1/monitoring**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        market_timezone: "America/New_York",
        window_start: "07:30",
        window_end: "18:00",
        evaluation_minutes: 30,
        risk_refresh_seconds: 60,
        enabled: false,
        consented: false,
        inside_window: true,
        is_trading_day: true,
        is_holiday: false,
        is_early_close: false,
        provider_status: "configured",
        running: false,
        notice: "Monitoring is disabled until you grant onboarding consent.",
      }),
    }),
  );
  await page.route("**/api/v1/portfolio**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        schema_version: 1,
        snapshot_id: "snapshot-browser-test",
        captured_at: "2026-07-11T16:30:00Z",
        state: "live",
        freshness: {
          as_of: "2026-07-11T16:30:00Z",
          provider: "tastytrade",
          state: "fresh",
        },
        accounts: [
          {
            account_id: "public-account-id",
            label: "Individual 1",
            account_type: "Individual",
            net_liquidating_value: 25000,
            cash_balance: 5000,
            buying_power: 10000,
            pnl_today: 0,
            positions: [],
          },
        ],
        strategies: [
          {
            strategy_id: "strat-browser",
            account_id: "public-account-id",
            underlying: "SPY",
            strategy_type: "Short Put",
            expiration_date: "2026-08-21",
            days_to_expiration: 21,
            quantity: 1,
            strikes: "$500",
            unrealized_pnl: 40,
            unrealized_pnl_percent: 10,
            total_delta: -20,
            total_theta: 4,
            horizon: "tactical",
            legs: [
              {
                symbol: "SPY  260821P00500000",
                underlying_symbol: "SPY",
                quantity: 1,
                quantity_direction: "Short",
                position_type: "Equity Option",
                strike_price: 500,
                option_type: "P",
                expiration_date: "2026-08-21",
                days_to_expiration: 21,
                mark_price: 2.1,
                market_value: -210,
                unrealized_pnl: 40,
                unrealized_pnl_percent: 10,
                delta: -0.2,
                gamma: 0.01,
                theta: 0.04,
                vega: -0.08,
                implied_volatility: 0.2,
                multiplier: 100,
                horizon: "tactical",
              },
            ],
          },
          {
            strategy_id: "strat-spy-stock",
            account_id: "public-account-id",
            underlying: "spy",
            strategy_type: "Long Stock",
            expiration_date: null,
            days_to_expiration: null,
            quantity: 100,
            strikes: "",
            unrealized_pnl: 120,
            unrealized_pnl_percent: 2.4,
            total_delta: 100,
            total_theta: 0,
            horizon: "strategic",
            legs: [
              {
                symbol: "SPY",
                underlying_symbol: "SPY",
                quantity: 100,
                quantity_direction: "Long",
                position_type: "Equity",
                strike_price: null,
                option_type: null,
                expiration_date: null,
                days_to_expiration: null,
                mark_price: 500,
                market_value: 50000,
                unrealized_pnl: 120,
                unrealized_pnl_percent: 2.4,
                delta: 1,
                gamma: 0,
                theta: 0,
                vega: 0,
                implied_volatility: null,
                multiplier: 1,
                horizon: "strategic",
              },
            ],
          },
          {
            strategy_id: "strat-qqq",
            account_id: "public-account-id",
            underlying: "QQQ",
            strategy_type: "Iron Condor",
            expiration_date: "2026-08-15",
            days_to_expiration: 18,
            quantity: 1,
            strikes: "$470 / $475 / $505 / $510",
            unrealized_pnl: -25,
            unrealized_pnl_percent: -8,
            total_delta: 2,
            total_theta: 12,
            horizon: "tactical",
            legs: [
              {
                symbol: "QQQ  260815P00470000",
                underlying_symbol: "QQQ",
                quantity: 1,
                quantity_direction: "Long",
                position_type: "Equity Option",
                strike_price: 470,
                option_type: "P",
                expiration_date: "2026-08-15",
                days_to_expiration: 18,
                mark_price: 0.4,
                market_value: 40,
                unrealized_pnl: -10,
                unrealized_pnl_percent: null,
                delta: 0.05,
                gamma: 0.01,
                theta: 0.02,
                vega: 0.03,
                implied_volatility: 0.18,
                multiplier: 100,
                horizon: "tactical",
              },
              {
                symbol: "QQQ  260815P00475000",
                underlying_symbol: "QQQ",
                quantity: 1,
                quantity_direction: "Short",
                position_type: "Equity Option",
                strike_price: 475,
                option_type: "P",
                expiration_date: "2026-08-15",
                days_to_expiration: 18,
                mark_price: 1.1,
                market_value: -110,
                unrealized_pnl: 15,
                unrealized_pnl_percent: null,
                delta: -0.12,
                gamma: 0.02,
                theta: 0.05,
                vega: -0.04,
                implied_volatility: 0.19,
                multiplier: 100,
                horizon: "tactical",
              },
              {
                symbol: "QQQ  260815C00505000",
                underlying_symbol: "QQQ",
                quantity: 1,
                quantity_direction: "Short",
                position_type: "Equity Option",
                strike_price: 505,
                option_type: "C",
                expiration_date: "2026-08-15",
                days_to_expiration: 18,
                mark_price: 1.2,
                market_value: -120,
                unrealized_pnl: 20,
                unrealized_pnl_percent: null,
                delta: -0.15,
                gamma: 0.02,
                theta: 0.06,
                vega: -0.05,
                implied_volatility: 0.17,
                multiplier: 100,
                horizon: "tactical",
              },
              {
                symbol: "QQQ  260815C00510000",
                underlying_symbol: "QQQ",
                quantity: 1,
                quantity_direction: "Long",
                position_type: "Equity Option",
                strike_price: 510,
                option_type: "C",
                expiration_date: "2026-08-15",
                days_to_expiration: 18,
                mark_price: 0.5,
                market_value: 50,
                unrealized_pnl: -50,
                unrealized_pnl_percent: null,
                delta: 0.08,
                gamma: 0.01,
                theta: 0.03,
                vega: 0.04,
                implied_volatility: 0.16,
                multiplier: 100,
                horizon: "tactical",
              },
            ],
          },
          {
            strategy_id: "strat-aapl-cc",
            account_id: "public-account-id",
            underlying: "AAPL",
            strategy_type: "Covered Call",
            expiration_date: "2026-08-21",
            days_to_expiration: 21,
            quantity: 1,
            strikes: "$220",
            unrealized_pnl: 55,
            unrealized_pnl_percent: 3,
            total_delta: 70,
            total_theta: 8,
            horizon: "tactical",
            legs: [
              {
                symbol: "AAPL",
                underlying_symbol: "AAPL",
                quantity: 100,
                quantity_direction: "Long",
                position_type: "Equity",
                strike_price: null,
                option_type: null,
                expiration_date: null,
                days_to_expiration: null,
                mark_price: 210,
                market_value: 21000,
                unrealized_pnl: 80,
                unrealized_pnl_percent: null,
                delta: 1,
                gamma: 0,
                theta: 0,
                vega: 0,
                implied_volatility: null,
                multiplier: 1,
                horizon: "tactical",
              },
              {
                symbol: "AAPL 260821C00220000",
                underlying_symbol: "AAPL",
                quantity: 1,
                quantity_direction: "Short",
                position_type: "Equity Option",
                strike_price: 220,
                option_type: "C",
                expiration_date: "2026-08-21",
                days_to_expiration: 21,
                mark_price: 1.5,
                market_value: -150,
                unrealized_pnl: -25,
                unrealized_pnl_percent: null,
                delta: -0.3,
                gamma: 0.02,
                theta: 0.08,
                vega: -0.05,
                implied_volatility: 0.22,
                multiplier: 100,
                horizon: "tactical",
              },
            ],
          },
        ],
        totals: {
          net_liquidating_value: 25000,
          cash_balance: 5000,
          buying_power: 10000,
          unrealized_pnl: 190,
        },
        selected_account_id: new URL(route.request().url()).searchParams.get("account_id") ?? "all",
        notice: null,
      }),
    }),
  );
  await page.route("**/api/v1/portfolio/risk**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        total_delta: -20,
        total_gamma: 0.1,
        total_theta: 4,
        total_vega: -8,
        unrealized_pnl: 40,
        net_liquidating_value: 25000,
        account_count: 1,
        strategy_count: 1,
        position_count: 1,
        concentration: [
          {
            underlying: "SPY",
            market_value: 250,
            share_of_portfolio: 0.01,
            strategy_count: 1,
            net_delta: -20,
          },
        ],
        stress: [
          {
            name: "theta_1d",
            label: "1-day theta",
            estimated_pnl_change: 4,
            description: "theta",
          },
        ],
      }),
    }),
  );
  await page.route("**/api/v1/markets", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        captured_at: "2026-07-11T16:30:00Z",
        quotes: [
          {
            symbol: "SPY",
            price: 500,
            bid: 499.9,
            ask: 500.1,
            iv: 0.16,
            iv_rank: 35,
            iv_percentile: 40,
            liquidity_rating: 4,
            iv_environment: "normal",
            spread_percent: 0.04,
          },
        ],
        iv_summary: { normal: 1 },
      }),
    }),
  );
  await page.route("**/api/v1/watchlist", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        symbols: ["SPY"],
        quotes: [
          {
            symbol: "SPY",
            price: 500,
            bid: 499.9,
            ask: 500.1,
            iv: 0.16,
            iv_rank: 35,
            iv_percentile: 40,
            liquidity_rating: 4,
            iv_environment: "normal",
            spread_percent: 0.04,
          },
        ],
      }),
    }),
  );
  await page.route("**/api/v1/streaming/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        market: { state: "disabled", error: null },
        account: { state: "disabled", error: null },
      }),
    }),
  );
  await page.route("**/api/v1/catalysts**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        captured_at: "2026-07-11T16:30:00Z",
        results: [
          {
            symbol: "SPY",
            confidence: "no_confirmed_catalyst_found",
            attribution: "none",
            summary: "No confirmed catalyst found",
            catalysts: [],
            option_mechanisms: [],
            social_side_notes: [],
            move_percent: 0.2,
            prior_close: 500,
            last_price: 501,
            meaningful_move: false,
            promoted: false,
            coverage: "complete",
            coverage_notes: [],
            quiet: true,
            freshness: {
              as_of: "2026-07-11T16:30:00Z",
              provider: "catalyst-service",
              state: "fresh",
            },
          },
        ],
        settings: {
          stock_move_threshold_pct: 2,
          etf_move_threshold_pct: 1,
          news_cadence_seconds: 300,
          benzinga: { enabled: false, status: "disabled" },
          scheduled_window_hours: 72,
        },
        coverage: "complete",
        coverage_notes: [],
        freshness: {
          as_of: "2026-07-11T16:30:00Z",
          provider: "catalyst-service",
          state: "fresh",
        },
      }),
    }),
  );
  await page.route("**/api/v1/settings/catalysts", (route) => {
    if (route.request().method() === "PUT") {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          stock_move_threshold_pct: 2,
          etf_move_threshold_pct: 1,
          news_cadence_seconds: 180,
          benzinga: { enabled: false, status: "disabled" },
          scheduled_window_hours: 72,
        }),
      });
    }
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        stock_move_threshold_pct: 2,
        etf_move_threshold_pct: 1,
        news_cadence_seconds: 300,
        benzinga: { enabled: false, status: "disabled" },
        scheduled_window_hours: 72,
      }),
    });
  });
  await page.route("**/api/v1/accounts/*/orders", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify([
        {
          order_id: "public-order",
          account_id: "public-account-id",
          symbol: "SPY 260821P00500000",
          underlying_symbol: "SPY",
          action: "Sell to Open",
          quantity: 1,
          order_type: "Limit",
          status: "filled",
          created_at: "2026-07-10T16:30:00Z",
          filled_quantity: 1,
          average_fill_price: 2.5,
          fills: [
            {
              fill_id: "public-fill",
              filled_at: "2026-07-10T16:31:00Z",
              symbol: "SPY",
              quantity: 1,
              price: 2.5,
              amount: 250,
            },
          ],
        },
      ]),
    }),
  );
  await page.route("**/api/v1/accounts/*/rolls/patterns**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        account_id: "public-account-id",
        symbol: null,
        avg_dte_at_roll: 14,
        typical_roll_days: [14],
        win_rate: 0.75,
        total_rolls: 4,
        avg_roll_pnl: 50,
        total_pnl: 200,
        best_dte_window: [10, 18],
        avg_days_between_rolls: 21,
      }),
    }),
  );
  await page.route("**/api/v1/accounts/*/rolls/heatmap**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        account_id: "public-account-id",
        underlying: "SPY",
        cells: [{ strike: 500, dte_bucket: "8-14", count: 1 }],
        strikes: [500],
        buckets: ["8-14"],
        total_rolls: 1,
      }),
    }),
  );
  const rollChain = {
    chain_id: "chain-public",
    account_id: "public-account-id",
    underlying: "SPY",
    strategy_type: "Short Put",
    original_open_credit: 2.5,
    chain_total_credit: 2.8,
    rolls: [
      {
        roll_id: "roll-public-older",
        timestamp: "2026-07-01T15:00:00Z",
        underlying: "SPY",
        strategy_type: "Short Put",
        old_symbol: "SPY  260717P00500000",
        old_strike: 500,
        old_expiration: "2026-07-17",
        old_dte: 14,
        new_symbol: "SPY  260731P00495000",
        new_strike: 495,
        new_expiration: "2026-07-31",
        new_dte: 28,
        old_quantity: 1,
        new_quantity: 1,
        roll_pnl: 40,
        premium_effect: 0.2,
        commission: 1,
        reason: null,
        notes: null,
      },
      {
        roll_id: "roll-public",
        timestamp: "2026-07-09T16:30:00Z",
        underlying: "SPY",
        strategy_type: "Short Put",
        old_symbol: "SPY  260731P00495000",
        old_strike: 495,
        old_expiration: "2026-07-31",
        old_dte: 14,
        new_symbol: "SPY  260821P00490000",
        new_strike: 490,
        new_expiration: "2026-08-21",
        new_dte: 35,
        old_quantity: 1,
        new_quantity: 1,
        roll_pnl: 50,
        premium_effect: 0.3,
        commission: 1,
        reason: null,
        notes: null,
      },
    ],
  };
  // Second chain with null lifetime total exercises partial lifetime credit honesty.
  const rollChainPartial = {
    chain_id: "chain-partial",
    account_id: "public-account-id",
    underlying: "QQQ",
    strategy_type: "Iron Condor",
    original_open_credit: null,
    chain_total_credit: null,
    rolls: [
      {
        roll_id: "roll-partial",
        timestamp: "2026-07-08T14:00:00Z",
        underlying: "QQQ",
        strategy_type: "Iron Condor",
        old_strike: 480,
        new_strike: 475,
        old_expiration: "2026-07-18",
        new_expiration: "2026-08-15",
        old_dte: 10,
        new_dte: 38,
        roll_pnl: -20,
        premium_effect: -0.15,
        commission: 2,
      },
    ],
  };
  await page.route("**/api/v1/accounts/*/rolls", (route) => {
    const url = route.request().url();
    // Account-scoped responses; default public account carries both chains.
    if (url.includes("/accounts/public-account-id/rolls")) {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify([rollChain, rollChainPartial]),
      });
    }
    if (url.includes("/accounts/second-account-id/rolls")) {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify([
          {
            chain_id: "chain-second",
            account_id: "second-account-id",
            underlying: "IWM",
            strategy_type: "Short Put",
            original_open_credit: 1.0,
            chain_total_credit: 1.4,
            rolls: [
              {
                roll_id: "roll-second",
                timestamp: "2026-07-11T12:00:00Z",
                underlying: "IWM",
                strategy_type: "Short Put",
                old_strike: 200,
                new_strike: 198,
                old_dte: 7,
                new_dte: 21,
                old_expiration: "2026-07-18",
                new_expiration: "2026-08-01",
                roll_pnl: 25,
                premium_effect: 0.4,
              },
            ],
          },
        ]),
      });
    }
    return route.fulfill({ contentType: "application/json", body: "[]" });
  });
  await page.route("**/api/v1/strategies/strat-browser", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        strategy: {
          strategy_id: "strat-browser",
          account_id: "public-account-id",
          underlying: "SPY",
          strategy_type: "Short Put",
          expiration_date: "2026-08-21",
          days_to_expiration: 21,
          quantity: 1,
          strikes: "$500",
          unrealized_pnl: 40,
          unrealized_pnl_percent: 10,
          total_delta: -20,
          total_theta: 4,
          horizon: "tactical",
          legs: [],
        },
        risk: {
          max_profit: 250,
          max_loss: 49750,
          breakevens: [497.5],
          distance_to_nearest_strike: 0,
          underlying_price: 500,
          current_pnl: 40,
          defined_risk: false,
          valuation_basis: "current_mark",
          combined: {
            delta: -20,
            gamma: 0.1,
            theta: 4,
            vega: -8,
            average_iv: 0.2,
            nearest_dte: 21,
          },
          stress: [
            {
              name: "theta_1d",
              label: "1-day theta",
              estimated_pnl_change: 4,
              description: "theta",
            },
          ],
        },
        market: {
          symbol: "SPY",
          price: 500,
          bid: 499.9,
          ask: 500.1,
          iv: 0.2,
          iv_rank: 35,
          iv_percentile: 40,
          liquidity_rating: 4,
          iv_environment: "normal",
          spread_percent: 0.04,
        },
        chart: {
          symbol: "SPY",
          bars: [
            {
              timestamp: "2026-07-11T16:30:00Z",
              open: 499,
              high: 501,
              low: 498,
              close: 500,
              volume: 1000,
            },
          ],
          source: "massive-stocks",
          notice: null,
          prior_close: 498,
          include_extended_hours: true,
          event_markers: [],
        },
        catalyst: {
          symbol: "SPY",
          confidence: "no_confirmed_catalyst_found",
          attribution: "none",
          summary: "No confirmed catalyst found",
          catalysts: [],
          option_mechanisms: [],
          social_side_notes: [],
          move_percent: 0.2,
          prior_close: 498,
          last_price: 500,
          meaningful_move: false,
          promoted: false,
          coverage: "complete",
          coverage_notes: [],
          quiet: true,
          freshness: {
            as_of: "2026-07-11T16:30:00Z",
            provider: "catalyst-service",
            state: "fresh",
          },
        },
        thesis: null,
        trade_plan: null,
        audit: [],
        rolls: [rollChain],
        events: [
          {
            kind: "roll",
            timestamp: "2026-07-09T16:30:00Z",
            summary: "Rolled 500 → 495",
            action: "roll",
          },
        ],
      }),
    }),
  );
  await page.route("**/api/v1/strategies/strat-aapl-cc", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        strategy: {
          strategy_id: "strat-aapl-cc",
          account_id: "public-account-id",
          underlying: "AAPL",
          strategy_type: "Covered Call",
          expiration_date: "2026-08-21",
          days_to_expiration: 21,
          quantity: 1,
          strikes: "$220",
          unrealized_pnl: 55,
          unrealized_pnl_percent: 3,
          total_delta: 70,
          total_theta: 8,
          horizon: "tactical",
          legs: [],
        },
        risk: {
          max_profit: 1000,
          max_loss: 21000,
          breakevens: [208.5],
          distance_to_nearest_strike: 10,
          underlying_price: 210,
          current_pnl: 55,
          defined_risk: true,
          valuation_basis: "current_mark",
          combined: {
            delta: 70,
            gamma: 0.02,
            theta: 8,
            vega: -5,
            average_iv: 0.22,
            nearest_dte: 21,
          },
          stress: [],
        },
        market: {
          symbol: "AAPL",
          price: 210,
          bid: 209.9,
          ask: 210.1,
          iv: 0.22,
          iv_rank: 40,
          iv_percentile: 45,
          liquidity_rating: 5,
          iv_environment: "normal",
          spread_percent: 0.05,
        },
        chart: {
          symbol: "AAPL",
          bars: [],
          source: "massive-stocks",
          notice: null,
          prior_close: 208,
          include_extended_hours: true,
          event_markers: [],
        },
        catalyst: null,
        thesis: null,
        trade_plan: null,
        audit: [],
        rolls: [],
        events: [],
      }),
    }),
  );

  // Phase 7 operations endpoints
  await page.route("**/api/v1/diagnostics/env", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        path: "/tmp/.env",
        exists: true,
        gitignored: true,
        tracked_by_git: false,
        permission_mode: "0o600",
        broadly_readable: false,
        warnings: [],
        note: "Credential values are never read or returned.",
      }),
    }),
  );
  await page.route("**/api/v1/diagnostics/bundle**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        generated_at: "2026-07-11T16:30:00Z",
        app_version: "0.1.0",
        schema_version: 6,
        provider_status: { tastytrade: "configured", codex: "configured" },
        settings_redacted: { theme: "dark" },
        env_diagnostics: {
          path: "/tmp/.env",
          exists: true,
          gitignored: true,
          tracked_by_git: false,
          permission_mode: "0o600",
          broadly_readable: false,
          warnings: [],
          note: "Credential values are never read or returned.",
        },
        monitoring: { active: false },
        counts: { portfolio_snapshots: 1 },
        redaction: {
          excluded: ["credential values", "prompts"],
          redacted_keys: [],
          policy: "redacted",
        },
        disclaimer: "decision support only",
      }),
    }),
  );
  await page.route("**/api/v1/settings/retention**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        portfolio_snapshots_days: 365,
        catalyst_events_days: 365,
        article_metadata_days: 90,
        recommendation_history_days: 0,
        transaction_history: "indefinite",
        candidates: {},
        audit_critical_preserved: ["recommendation history (indefinite)"],
        would_delete: {},
        disclaimer: "decision support only",
      }),
    }),
  );
  await page.route("**/api/v1/backups**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: route.request().method() === "POST"
        ? JSON.stringify({
            backup_id: "position-pilot-manual-test.sqlite3",
            filename: "position-pilot-manual-test.sqlite3",
            path: "/tmp/position-pilot-manual-test.sqlite3",
            size_bytes: 1024,
            created_at: "2026-07-11T16:30:00Z",
            reason: "manual",
            schema_version: 6,
            app_version: "0.1.0",
            sha256: "abc",
            integrity_ok: true,
          })
        : "[]",
    }),
  );
  await page.route("**/api/v1/update/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        current_version: "0.1.0",
        latest_version: null,
        update_available: false,
        schema_version: 6,
        schema_migrations_pending: false,
        backup_required_before_update: true,
        monitoring_active: false,
        blocked_reason: null,
        reversible_instructions: ["1. Disable monitoring", "2. Create backup"],
        auto_install: false,
        note: "Updates are never installed automatically.",
        disclaimer: "decision support only",
      }),
    }),
  );
  await page.route("**/api/v1/exports/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/plain",
      headers: { "Content-Disposition": 'attachment; filename="export-test.txt"' },
      body: "export-ok",
    }),
  );
}

test("secure dashboard shell renders without console or accessibility errors", async ({
  page,
}) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  await mockDashboardApis(page);

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);

  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();
  await expect(page).not.toHaveURL(/launch_token/);
  await expect(page.getByRole("complementary", { name: "Primary navigation" })).toBeVisible();
  await expect(page.getByText("Provider ledger")).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Account scope" })).toBeVisible();
  await expect(page.getByText("$25,000.00").first()).toBeVisible();
  await expect(page.getByText("Catalyst intelligence.")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Held-symbol catalysts" })).toBeVisible();

  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  await expect(page.getByText("Short Put").first()).toBeVisible();
  await expect(page.getByText("No confirmed catalyst found").first()).toBeVisible();
  // Symbol groups: alphabetical AAPL, QQQ, SPY; SPY holds stock + option under one disclosure.
  await expect(page.getByRole("button", { name: /SPY/i }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: /AAPL/i }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: /QQQ/i }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toBeVisible();
  // Combined multi-leg rows use a distinct analysis action (not the disclosure toggle).
  await expect(
    page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Open analysis for QQQ Iron Condor" }),
  ).toBeVisible();
  await expect(page.getByRole("checkbox", { name: /Show stock positions/i })).toBeChecked();
  await expect(page.getByRole("checkbox", { name: /Show options positions/i })).toBeChecked();
  await expect(page.getByRole("heading", { name: "Order activity" })).toBeVisible();
  // Roll ledger rail sits beside Positions with flattened events (newest first).
  await expect(page.getByRole("heading", { name: "Roll ledger" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Option roll event ledger" })).toBeVisible();
  await expect(page.getByText("Net roll credit")).toBeVisible();
  // Net: 0.2 + 0.3 + (-0.15) = 0.35 → +$0.35 CR (excludes opening credit)
  await expect(page.getByText("+$0.35").first()).toBeVisible();
  await expect(page.getByText("Known lifetime credit")).toBeVisible();
  await expect(page.getByText("Partial")).toBeVisible();
  await expect(page.getByText(/lifetime total is incomplete/i)).toBeVisible();
  const ledgerRows = page.locator(".roll-ledger-row");
  await expect(ledgerRows).toHaveCount(3);
  // Newest first: Jul 9 SPY 495→490, then Jul 8 QQQ, then Jul 1 SPY 500→495
  await expect(ledgerRows.nth(0)).toContainText("SPY");
  await expect(ledgerRows.nth(0)).toContainText("$495");
  await expect(ledgerRows.nth(0)).toContainText("$490");
  await expect(ledgerRows.nth(1)).toContainText("QQQ");
  await expect(ledgerRows.nth(2)).toContainText("$500");
  await expect(page.getByRole("button", { name: "Open advanced Roll analytics" })).toBeVisible();

  await page.getByRole("button", { name: "Open SPY Short Put" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByText("Remaining max profit")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Catalysts" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Close", exact: true })).toBeFocused();
  await expect(page.getByLabel("Decision horizon")).toHaveValue("tactical");
  await expect(page.getByLabel("Profit target")).toBeVisible();
  await expect(page.getByLabel("Roll criteria")).toBeVisible();
  const drawerAccessibility = await new AxeBuilder({ page }).analyze();
  expect(drawerAccessibility.violations).toEqual([]);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();

  // Compact affordance opens advanced Roll analytics without inventing URL routing.
  await page.getByRole("button", { name: "Open advanced Roll analytics" }).click();
  await expect(page.getByRole("heading", { name: "Roll chains" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Pattern analytics" })).toBeVisible();

  await page.getByRole("button", { name: "Roll analytics" }).click();
  await expect(page.getByRole("heading", { name: "Roll chains" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Pattern analytics" })).toBeVisible();

  await page.getByRole("button", { name: "Markets" }).click();
  await expect(page.getByRole("heading", { name: "Market overview" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Quotes" })).toBeVisible();

  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("heading", { name: "Saved identities" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "News cadence and thresholds" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Portfolio and history" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Redacted bundle and .env checks" }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Create, download, restore" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Readiness (never auto-installed)" }),
  ).toBeVisible();
  await expect(page.getByText("decision support only", { exact: false }).first()).toBeVisible();

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test("symbol-grouped positions: collapse, category toggles, ledger isolation, overflow, a11y", async ({
  page,
}) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  // Deterministic alphabetical group order and same-symbol stock+option co-location.
  const groupToggles = page.locator("button.symbol-group-toggle");
  await expect(groupToggles).toHaveCount(3);
  await expect(groupToggles.nth(0)).toContainText("AAPL");
  await expect(groupToggles.nth(1)).toContainText("QQQ");
  await expect(groupToggles.nth(2)).toContainText("SPY");
  await expect(groupToggles.nth(2)).toContainText("2 positions");
  await expect(groupToggles.nth(2)).toContainText("1 stock");
  await expect(groupToggles.nth(2)).toContainText("1 option");
  await expect(groupToggles.nth(2)).not.toContainText("1 options");
  await expect(groupToggles.nth(2)).toHaveAttribute("aria-expanded", "true");
  await expect(groupToggles.nth(2)).toHaveAttribute("aria-controls", "symbol-group-panel-SPY");
  await expect(page.locator("th[scope='rowgroup']")).toHaveCount(3);
  await expect(
    page.locator("th[scope='rowgroup']").filter({ hasText: "SPY" }),
  ).toBeVisible();

  // Combined multi-leg strategies render as one strategy row each with legs always visible.
  await expect(
    page.locator("tr[data-level='1'].combined-strategy-row").filter({ hasText: "Iron Condor" }),
  ).toBeVisible();
  await expect(
    page.locator("tr[data-level='1'].combined-strategy-row").filter({ hasText: "Covered Call" }),
  ).toBeVisible();
  await expect(page.getByText("4 legs").first()).toBeVisible();
  await expect(page.getByText("2 legs").first()).toBeVisible();
  await expect(page.getByRole("button", { name: /Expand legs|Collapse legs/i })).toHaveCount(0);
  await expect(page.locator(".strategy-legs-toggle")).toHaveCount(0);
  await expect(page.locator(".strategy-legs-chevron")).toHaveCount(0);
  // Legs for combined strategies are visible under expanded symbols by default.
  await expect(
    page.locator("tr[data-level='2'][data-parent-strategy='strat-qqq']:visible"),
  ).toHaveCount(4);
  await expect(
    page.locator("tr[data-level='2'][data-parent-strategy='strat-aapl-cc']:visible"),
  ).toHaveCount(2);
  // Single-leg rows open analysis directly and have no leg disclosure.
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toBeVisible();
  await expect(
    page.getByText(/Multi-leg rows are Position Pilot detected strategies/i),
  ).toBeVisible();
  await expect(page.getByText(/individual legs are shown below/i)).toBeVisible();

  // Collapse only SPY; AAPL/QQQ rows remain visible.
  await groupToggles.nth(2).click();
  await expect(groupToggles.nth(2)).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Open analysis for QQQ Iron Condor" }),
  ).toBeVisible();

  // Re-expand SPY.
  await groupToggles.nth(2).click();
  await expect(groupToggles.nth(2)).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toBeVisible();

  const stockToggle = page.getByRole("checkbox", { name: /Show stock positions/i });
  const optionsToggle = page.getByRole("checkbox", { name: /Show options positions/i });
  const ledgerRows = page.locator(".roll-ledger-row");
  await expect(ledgerRows).toHaveCount(3);

  // Stock off: hide Long Stock, keep option/mixed; ledger unchanged.
  await stockToggle.uncheck();
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Open analysis for QQQ Iron Condor" }),
  ).toBeVisible();
  await expect(ledgerRows).toHaveCount(3);
  await expect(page.getByText("+$0.35").first()).toBeVisible();

  // Collapse state for SPY preserved while filtering.
  await groupToggles.filter({ hasText: "SPY" }).click();
  await expect(groupToggles.filter({ hasText: "SPY" })).toHaveAttribute("aria-expanded", "false");
  await stockToggle.check();
  await expect(groupToggles.filter({ hasText: "SPY" })).toHaveAttribute("aria-expanded", "false");
  await groupToggles.filter({ hasText: "SPY" }).click();

  // Options off: hide option/mixed, keep stock-only; ledger unchanged.
  await optionsToggle.uncheck();
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Open analysis for QQQ Iron Condor" }),
  ).toHaveCount(0);
  await expect(ledgerRows).toHaveCount(3);

  // Both off → filtered empty state with recovery actions; ledger still present.
  await stockToggle.uncheck();
  await expect(page.getByText(/All position categories are hidden/i)).toBeVisible();
  await expect(page.getByRole("button", { name: "Show Stock" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Show Options" })).toBeVisible();
  await expect(ledgerRows).toHaveCount(3);
  await page.getByRole("button", { name: "Show Options" }).click();
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toBeVisible();
  await expect(optionsToggle).toBeChecked();

  // No document horizontal overflow at desktop and phone widths on Positions.
  for (const size of [
    { width: 1280, height: 800 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(size);
    await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
    const overflow = await page.evaluate(() => {
      const doc = document.documentElement;
      return {
        scrollWidth: doc.scrollWidth,
        clientWidth: doc.clientWidth,
      };
    });
    expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth + 1);
  }

  await page.setViewportSize({ width: 1280, height: 800 });
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("positions hierarchy levels, aligned leg rows, contract identity, and P/L bars", async ({
  page,
}) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  // Hierarchy hooks: symbol (0), strategy (1), and always-visible combined legs (2).
  const level0 = page.locator("tr[data-level='0']");
  const level1 = page.locator("tr[data-level='1']");
  await expect(level0).toHaveCount(3);
  await expect(level1).toHaveCount(4);
  await expect(page.locator("tr.position-row-level-0")).toHaveCount(3);
  await expect(page.locator("tr.position-row-level-1")).toHaveCount(4);
  // Combined strategies always show legs under expanded symbols (IC 4 + CC 2).
  await expect(page.locator("tr[data-level='2']:visible")).toHaveCount(6);
  await expect(page.getByRole("button", { name: /Expand legs|Collapse legs/i })).toHaveCount(0);

  // Explicit colgroup / column width system + numeric alignment hooks.
  await expect(page.locator("table.positions-by-symbol > colgroup > col")).toHaveCount(8);
  await expect(page.locator("th.positions-th-num")).toHaveCount(5);
  await expect(page.locator("th.positions-th-position")).toBeVisible();
  await expect(page.locator("th.positions-th-pnl")).toHaveClass(/positions-th-num/);

  // Level-0: identity spans first columns; aggregate P/L lives in the global P/L column.
  const level0Structure = await page.evaluate(() => {
    const row = document.querySelector<HTMLTableRowElement>("tr[data-level='0']");
    if (!row) return null;
    const cells = Array.from(row.querySelectorAll(":scope > th, :scope > td"));
    const identity = cells[0] as HTMLTableCellElement | undefined;
    const pnl = cells[cells.length - 1] as HTMLTableCellElement | undefined;
    const colSpanTotal = cells.reduce(
      (sum, cell) => sum + ((cell as HTMLTableCellElement).colSpan || 1),
      0,
    );
    return {
      cellCount: cells.length,
      colSpanTotal,
      identityColSpan: identity?.colSpan ?? 0,
      hasAggregateClass: pnl?.classList.contains("symbol-group-aggregate-pnl") ?? false,
      pnlText: (pnl?.textContent || "").replace(/\s+/g, " ").trim(),
      identityHasPnlClass: identity?.querySelector(".symbol-group-pnl") != null,
    };
  });
  expect(level0Structure).not.toBeNull();
  expect(level0Structure!.colSpanTotal).toBe(8);
  expect(level0Structure!.identityColSpan).toBe(7);
  expect(level0Structure!.hasAggregateClass).toBeTruthy();
  expect(level0Structure!.identityHasPnlClass).toBeFalsy();
  expect(level0Structure!.pnlText.length).toBeGreaterThan(0);

  // Indentation hooks: level-1 exceeds level-0; level-2 matches level-1 (no extra x-indent).
  const paddingLeft = await page.evaluate(() => {
    const l0 = document.querySelector<HTMLElement>(
      "tr[data-level='0'] th, tr[data-level='0'] td",
    );
    const l1 = document.querySelector<HTMLElement>("tr[data-level='1'] th[scope='row']");
    const l2 = document.querySelector<HTMLElement>("tr[data-level='2'] th[scope='row']");
    return {
      l0: l0 ? Number.parseFloat(getComputedStyle(l0).paddingLeft) : -1,
      l1: l1 ? Number.parseFloat(getComputedStyle(l1).paddingLeft) : -1,
      l2: l2 ? Number.parseFloat(getComputedStyle(l2).paddingLeft) : -1,
    };
  });
  expect(paddingLeft.l1).toBeGreaterThan(paddingLeft.l0);
  expect(Math.abs(paddingLeft.l2 - paddingLeft.l1)).toBeLessThanOrEqual(1);

  // Level-0 magenta position dots (reference-style anchors) on every symbol row.
  await expect(page.locator("tr[data-level='0'] .symbol-group-dot")).toHaveCount(3);
  await expect(
    page.locator("tr[data-level='0'] .symbol-group-dot[aria-hidden='true']"),
  ).toHaveCount(3);

  // Magenta hierarchy dots present; old gold vertical guide hooks removed.
  await expect(page.locator("tr[data-level='1'] .position-hierarchy-dot")).toHaveCount(4);
  await expect(page.locator(".position-hierarchy-guide")).toHaveCount(0);

  // Aggregate symbol P/L uses signed tone classes (currency text remains primary signal).
  // Fixtures: AAPL +$55, QQQ −$25, SPY +$160.
  const aaplHeader = page.locator("tr[data-level='0']").filter({ hasText: "AAPL" });
  const qqqHeader = page.locator("tr[data-level='0']").filter({ hasText: "QQQ" });
  const spyHeader = page.locator("tr[data-level='0']").filter({ hasText: "SPY" });
  await expect(aaplHeader.locator(".symbol-group-pnl")).toHaveClass(
    /symbol-group-pnl-positive/,
  );
  await expect(aaplHeader.locator(".symbol-group-pnl")).toContainText("$55");
  await expect(qqqHeader.locator(".symbol-group-pnl")).toHaveClass(
    /symbol-group-pnl-negative/,
  );
  await expect(qqqHeader.locator(".symbol-group-pnl")).toContainText("$25");
  await expect(spyHeader.locator(".symbol-group-pnl")).toHaveClass(
    /symbol-group-pnl-positive/,
  );

  // One global header — no nested strategy-legs table.
  await expect(page.locator("table.positions-by-symbol > thead")).toHaveCount(1);
  await expect(page.locator("table.strategy-legs-table")).toHaveCount(0);
  await expect(page.locator(".strategy-legs-detail-row")).toHaveCount(0);

  // Positive / negative / unavailable unrealized P/L percent on strategy rows.
  const spyPutRow = page
    .locator("tr[data-level='1']")
    .filter({ has: page.getByRole("button", { name: "Open SPY Short Put" }) });
  await expect(spyPutRow.locator(".pnl-metric")).toContainText("$40.00");
  await expect(spyPutRow.locator(".pnl-metric")).toHaveClass(/pnl-metric-positive/);
  await expect(spyPutRow.locator(".pnl-metric-percent")).toContainText("+10.0%");
  await expect(spyPutRow.locator(".pnl-bar-fill-positive")).toBeVisible();
  await expect(spyPutRow.locator(".positions-td-pnl")).toHaveCount(1);

  const qqqStrategyRow = page
    .locator("tr[data-level='1'].combined-strategy-row")
    .filter({ hasText: "Iron Condor" });
  await expect(qqqStrategyRow.locator(".pnl-metric")).toContainText("$25.00");
  await expect(qqqStrategyRow.locator(".pnl-metric")).toHaveClass(/pnl-metric-negative/);
  await expect(qqqStrategyRow.locator(".pnl-metric-percent")).toContainText("-8.0%");
  await expect(qqqStrategyRow.locator(".pnl-bar-fill-negative")).toBeVisible();

  // Dense strategy row: name + leg count + analyze on one line; no disclosure control.
  await expect(qqqStrategyRow.locator(".strategy-label-row")).toHaveCount(1);
  await expect(qqqStrategyRow.locator(".strategy-label-stack")).toHaveCount(0);
  await expect(qqqStrategyRow.locator(".strategy-combined-name")).toHaveText("Iron Condor");
  await expect(qqqStrategyRow.locator(".strategy-leg-count")).toHaveText("4 legs");
  await expect(
    qqqStrategyRow.getByRole("button", { name: "Open analysis for QQQ Iron Condor" }),
  ).toHaveText("Analyze");
  const strategyRowLayout = await qqqStrategyRow.evaluate((row) => {
    const labelRow = row.querySelector(".strategy-label-row") as HTMLElement | null;
    if (!labelRow) return { ok: false };
    const style = getComputedStyle(labelRow);
    return {
      ok: true,
      display: style.display,
      flexDirection: style.flexDirection,
      hasToggle: labelRow.querySelector(".strategy-legs-toggle") != null,
      hasChevron: labelRow.querySelector(".strategy-legs-chevron") != null,
      hasName: labelRow.querySelector(".strategy-combined-name") != null,
      hasAnalyze: labelRow.querySelector(".strategy-analysis-action") != null,
    };
  });
  expect(strategyRowLayout.ok).toBeTruthy();
  expect(strategyRowLayout.display).toBe("flex");
  expect(strategyRowLayout.flexDirection).toBe("row");
  expect(strategyRowLayout.hasToggle).toBeFalsy();
  expect(strategyRowLayout.hasChevron).toBeFalsy();
  expect(strategyRowLayout.hasName).toBeTruthy();
  expect(strategyRowLayout.hasAnalyze).toBeTruthy();

  // Combined legs always visible: four aligned main-table leg rows; no nested table/header.
  const qqqLegs = page.locator(
    "tr[data-level='2'][data-parent-strategy='strat-qqq']:visible",
  );
  await expect(qqqLegs).toHaveCount(4);
  await expect(page.locator("table.strategy-legs-table")).toHaveCount(0);
  await expect(page.locator("table.positions-by-symbol thead")).toHaveCount(1);
  await expect(
    page.locator("tr[data-level='2']:visible .position-hierarchy-dot"),
  ).toHaveCount(6);

  // Rendered hierarchy alignment (desktop): measure painted left edges after gaps.
  // Level-0 → level-1 still steps ≥1rem. Level-2 leg dots/strips align with
  // single-leg strategy dots/strips (≤1px). Combined parent matches single strategy.
  await page.setViewportSize({ width: 1280, height: 800 });
  const visualHierarchy = await page.evaluate(() => {
    const rem = Number.parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    const minStep = rem * 0.95; // allow tiny subpixel tolerance under 1rem target
    const qqqSymbolRow = Array.from(
      document.querySelectorAll<HTMLElement>("tr[data-level='0']"),
    ).find((row) => row.querySelector(".symbol-group-symbol")?.textContent?.trim() === "QQQ");
    const qqqStrategyRow = Array.from(
      document.querySelectorAll<HTMLElement>("tr[data-level='1'].combined-strategy-row"),
    ).find((row) => (row.textContent || "").includes("Iron Condor"));
    const spyPutRow = Array.from(
      document.querySelectorAll<HTMLElement>("tr[data-level='1']"),
    ).find((row) => {
      const open = row.querySelector(".strategy-open-action");
      return (open?.textContent || "").includes("Short Put");
    });
    const qqqLegRow = document.querySelector<HTMLElement>(
      "tr[data-level='2'][data-parent-strategy='strat-qqq']",
    );
    const symbolLabel = qqqSymbolRow?.querySelector<HTMLElement>(".symbol-group-symbol");
    const l1Dot = qqqStrategyRow?.querySelector<HTMLElement>(".position-hierarchy-dot");
    const l1Name = qqqStrategyRow?.querySelector<HTMLElement>(".strategy-combined-name");
    const singleDot = spyPutRow?.querySelector<HTMLElement>(".position-hierarchy-dot");
    const singleName = spyPutRow?.querySelector<HTMLElement>(".strategy-open-action");
    const singleStrip = spyPutRow?.querySelector<HTMLElement>(".contract-strip");
    const l2Dot = qqqLegRow?.querySelector<HTMLElement>(".position-hierarchy-dot");
    const l2Strip = qqqLegRow?.querySelector<HTMLElement>(".contract-strip");
    if (
      !symbolLabel ||
      !l1Dot ||
      !l1Name ||
      !singleDot ||
      !singleName ||
      !singleStrip ||
      !l2Dot ||
      !l2Strip
    ) {
      return { ok: false as const, rem, minStep };
    }
    const symbolLeft = symbolLabel.getBoundingClientRect().left;
    const l1DotLeft = l1Dot.getBoundingClientRect().left;
    const l1NameLeft = l1Name.getBoundingClientRect().left;
    const singleDotLeft = singleDot.getBoundingClientRect().left;
    const singleNameLeft = singleName.getBoundingClientRect().left;
    const singleStripLeft = singleStrip.getBoundingClientRect().left;
    const l2DotLeft = l2Dot.getBoundingClientRect().left;
    const l2StripLeft = l2Strip.getBoundingClientRect().left;
    return {
      ok: true as const,
      rem,
      minStep,
      symbolLeft,
      l1DotLeft,
      l1NameLeft,
      singleDotLeft,
      singleNameLeft,
      singleStripLeft,
      l2DotLeft,
      l2StripLeft,
      step1: l1DotLeft - symbolLeft,
      step1Name: l1NameLeft - symbolLeft,
      legVsSingleDotDelta: Math.abs(l2DotLeft - singleDotLeft),
      legVsSingleStripDelta: Math.abs(l2StripLeft - singleStripLeft),
      combinedVsSingleDotDelta: Math.abs(l1DotLeft - singleDotLeft),
      combinedVsSingleNameDelta: Math.abs(l1NameLeft - singleNameLeft),
    };
  });
  expect(visualHierarchy.ok).toBeTruthy();
  if (visualHierarchy.ok) {
    expect(visualHierarchy.step1).toBeGreaterThanOrEqual(visualHierarchy.minStep);
    expect(visualHierarchy.step1Name).toBeGreaterThanOrEqual(visualHierarchy.minStep);
    expect(visualHierarchy.legVsSingleDotDelta).toBeLessThanOrEqual(1);
    expect(visualHierarchy.legVsSingleStripDelta).toBeLessThanOrEqual(1);
    expect(visualHierarchy.combinedVsSingleDotDelta).toBeLessThanOrEqual(1);
    expect(visualHierarchy.combinedVsSingleNameDelta).toBeLessThanOrEqual(1);
  }

  // Joined contract strip: signed qty, expiry, DTE, strike, option type — no · separators.
  const longPutLeg = qqqLegs.filter({ hasText: "$470" }).first();
  await expect(longPutLeg.locator(".contract-strip")).toHaveCount(1);
  await expect(longPutLeg.locator(".contract-qty")).toHaveText("+1");
  await expect(longPutLeg.locator(".contract-exp")).toContainText("Aug");
  await expect(longPutLeg.locator(".contract-dte")).toHaveText("18d");
  await expect(longPutLeg.locator(".contract-strike")).toHaveText("$470");
  await expect(longPutLeg.locator(".contract-type")).toHaveText("P");
  await expect(longPutLeg.locator(".sr-only")).toContainText("Long 1");
  await expect(longPutLeg.locator(".contract-seg-sep")).toHaveCount(0);
  await expect(longPutLeg.locator(".contract-strip")).not.toContainText("·");

  const shortCallLeg = qqqLegs.filter({ hasText: "$505" }).first();
  await expect(shortCallLeg.locator(".contract-qty")).toHaveText("−1");
  await expect(shortCallLeg.locator(".contract-type")).toHaveText("C");
  await expect(shortCallLeg.locator(".contract-strike")).toHaveText("$505");

  // Unavailable leg percent: value shown, bar omitted.
  await expect(longPutLeg.locator(".pnl-metric-value")).toBeVisible();
  await expect(longPutLeg.locator(".pnl-bar-track")).toHaveCount(0);
  await expect(longPutLeg.locator(".pnl-metric-percent")).toHaveCount(0);

  // Level-2 shares level-1 left inset (no extra x-indent for combined legs).
  const legPad = await page.evaluate(() => {
    const l1 = document.querySelector<HTMLElement>("tr[data-level='1'] th[scope='row']");
    const l2 = document.querySelector<HTMLElement>(
      "tr[data-level='2'] th[scope='row']",
    );
    return {
      l1: l1 ? Number.parseFloat(getComputedStyle(l1).paddingLeft) : -1,
      l2: l2 ? Number.parseFloat(getComputedStyle(l2).paddingLeft) : -1,
    };
  });
  expect(Math.abs(legPad.l2 - legPad.l1)).toBeLessThanOrEqual(1);

  // Equity leg identity on Covered Call (always visible).
  const aaplLegs = page.locator(
    "tr[data-level='2'][data-parent-strategy='strat-aapl-cc']:visible",
  );
  await expect(aaplLegs).toHaveCount(2);
  const equityLeg = aaplLegs.filter({ hasText: "Equity" }).first();
  await expect(equityLeg.locator(".contract-qty")).toHaveText("+100");
  await expect(equityLeg.locator(".contract-instrument")).toHaveText("Equity");
  await expect(equityLeg.locator(".contract-strip")).toHaveCount(1);
  const callLeg = aaplLegs.filter({ hasText: "$220" }).first();
  await expect(callLeg.locator(".contract-qty")).toHaveText("−1");
  await expect(callLeg.locator(".contract-type")).toHaveText("C");
  await expect(callLeg.locator(".contract-strike")).toHaveText("$220");

  // Single-leg strategy also shows compact identity (not a combined disclosure).
  const spyPutIdentity = spyPutRow.locator(".contract-identity");
  await expect(spyPutIdentity.locator(".contract-qty")).toHaveText("−1");
  await expect(spyPutIdentity.locator(".contract-type")).toHaveText("P");
  await expect(spyPutIdentity.locator(".contract-strike")).toHaveText("$500");

  // Within SPY: shares/stock row precedes options even though options arrive first in fixture.
  const spyOrder = await page.evaluate(() => {
    const spyPanel = document.getElementById("symbol-group-panel-SPY");
    if (!spyPanel) return [] as string[];
    return Array.from(spyPanel.querySelectorAll<HTMLElement>("tr[data-level='1']")).map(
      (row) => {
        const open = row.querySelector<HTMLElement>(".strategy-open-action");
        return (open?.textContent || row.textContent || "").replace(/\s+/g, " ").trim();
      },
    );
  });
  expect(spyOrder.length).toBe(2);
  expect(spyOrder[0]).toMatch(/Long Stock/i);
  expect(spyOrder[1]).toMatch(/Short Put/i);

  // Fixed-track instrument grid: option legs always expose 5 segments; equity uses same width.
  await expect(longPutLeg.locator(".contract-grid .contract-seg")).toHaveCount(5);
  await expect(shortCallLeg.locator(".contract-grid .contract-seg")).toHaveCount(5);
  await expect(spyPutIdentity.locator(".contract-grid .contract-seg")).toHaveCount(5);
  await expect(equityLeg.locator(".contract-identity-equity .contract-grid")).toHaveCount(1);
  await expect(equityLeg.locator(".contract-grid .contract-seg")).toHaveCount(2);

  const gridAlignment = await page.evaluate(() => {
    const optionStrips = Array.from(
      document.querySelectorAll<HTMLElement>(
        "tr[data-level='2']:not([hidden]) .contract-identity-option .contract-grid",
      ),
    );
    const equityStrip = document.querySelector<HTMLElement>(
      "tr[data-level='2']:not([hidden]) .contract-identity-equity .contract-grid",
    );
    if (optionStrips.length < 2 || !equityStrip) {
      return { ok: false as const };
    }
    const widths = optionStrips.map((el) => Math.round(el.getBoundingClientRect().width));
    const lefts = optionStrips.map((el) => Math.round(el.getBoundingClientRect().left));
    const equityWidth = Math.round(equityStrip.getBoundingClientRect().width);
    const equityLeft = Math.round(equityStrip.getBoundingClientRect().left);

    // Column track positions for qty / strike across two option legs with different content.
    const trackLefts = optionStrips.map((strip) => {
      const qty = strip.querySelector<HTMLElement>(".contract-qty");
      const strike = strip.querySelector<HTMLElement>(".contract-strike");
      const type = strip.querySelector<HTMLElement>(".contract-type");
      return {
        qty: qty ? Math.round(qty.getBoundingClientRect().left) : -1,
        strike: strike ? Math.round(strike.getBoundingClientRect().left) : -1,
        type: type ? Math.round(type.getBoundingClientRect().left) : -1,
      };
    });

    const first = trackLefts[0]!;
    const tracksAligned = trackLefts.every(
      (t) =>
        t.qty === first.qty && t.strike === first.strike && t.type === first.type,
    );

    return {
      ok: true as const,
      widths,
      lefts,
      equityWidth,
      equityLeft,
      optionWidth: widths[0] ?? 0,
      tracksAligned,
      display: getComputedStyle(optionStrips[0]!).display,
    };
  });
  expect(gridAlignment.ok).toBeTruthy();
  if (gridAlignment.ok) {
    expect(gridAlignment.display).toBe("grid");
    expect(new Set(gridAlignment.widths).size).toBe(1);
    expect(gridAlignment.equityWidth).toBe(gridAlignment.optionWidth);
    expect(gridAlignment.tracksAligned).toBeTruthy();
    // Equity and option descriptors share the same left origin within their respective cells;
    // compare width equality only here (indent levels differ for nested legs).
    expect(gridAlignment.optionWidth).toBeGreaterThan(80);
  }

  // Strategic vs Tactical horizon chips are visually distinct (not identical graphite).
  const horizonColors = await page.evaluate(() => {
    const strategic = document.querySelector<HTMLElement>(
      "tr[data-level='1'] .horizon-strategic",
    );
    const tactical = document.querySelector<HTMLElement>(
      "tr[data-level='1'] .horizon-tactical",
    );
    if (!strategic || !tactical) return null;
    const s = getComputedStyle(strategic);
    const t = getComputedStyle(tactical);
    return {
      strategicColor: s.color,
      tacticalColor: t.color,
      strategicBorder: s.borderColor,
      tacticalBorder: t.borderColor,
      strategicBg: s.backgroundColor,
      tacticalBg: t.backgroundColor,
    };
  });
  expect(horizonColors).not.toBeNull();
  expect(horizonColors!.strategicColor).not.toBe(horizonColors!.tacticalColor);

  // Dark theme palette: neutral charcoal bands + magenta dots + Tastytrade-like P/L.
  // Prefer computed styles / CSS tokens over brittle screenshot snapshots.
  await expect
    .poll(async () =>
      page.evaluate(() => document.documentElement.getAttribute("data-theme")),
    )
    .not.toBe("light");
  // Clear disclosure focus so strategy-row sampling is not magenta-tinted by :focus-within.
  await page.evaluate(() => {
    const active = document.activeElement;
    if (active instanceof HTMLElement) active.blur();
  });
  const darkPalette = await page.evaluate(() => {
    type Oklch = { l: number; c: number; h: number };
    const parseOklch = (value: string): Oklch | null => {
      const m = value.match(
        /oklch\(\s*([\d.]+)\s*%?\s+([\d.]+)\s+(-?[\d.]+)/i,
      );
      if (!m) return null;
      let l = Number(m[1]);
      // Browsers may serialize as 0–1 or 0–100%.
      if (l <= 1) l *= 100;
      return { l, c: Number(m[2]), h: Number(m[3]) };
    };
    const parseTokenOklch = (value: string): Oklch | null => parseOklch(value);

    const table = document.querySelector<HTMLElement>("table.positions-by-symbol");
    const header = document.querySelector<HTMLElement>("table.positions-by-symbol thead th");
    const symbolCell = document.querySelector<HTMLElement>(
      "tr[data-level='0'] th, tr[data-level='0'] td",
    );
    // Prefer a strategy row that is not focus-within (disclosure blur above).
    const strategyCell = document.querySelector<HTMLElement>(
      "tr[data-level='1']:not(:focus-within) th[scope='row']",
    ) ?? document.querySelector<HTMLElement>("tr[data-level='1'] th[scope='row']");
    const legCell = document.querySelector<HTMLElement>(
      "tr[data-level='2']:not([hidden]) th[scope='row']",
    );
    const contractStrip = document.querySelector<HTMLElement>(
      "tr[data-level='2']:not([hidden]) .contract-strip",
    );
    const dot = document.querySelector<HTMLElement>(
      "tr[data-level='0'] .position-hierarchy-dot",
    );
    const hierarchyDot = document.querySelector<HTMLElement>(
      "tr[data-level='1'] .position-hierarchy-dot",
    );
    const positivePnl = document.querySelector<HTMLElement>(
      ".symbol-group-pnl-positive, .pnl-metric-positive .pnl-metric-value",
    );
    const negativePnl = document.querySelector<HTMLElement>(
      ".symbol-group-pnl-negative, .pnl-metric-negative .pnl-metric-value",
    );
    const symbolLabel = document.querySelector<HTMLElement>(".symbol-group-symbol");
    const singleStrategyName = document.querySelector<HTMLElement>("button.strategy-open-action");
    const combinedStrategyName = document.querySelector<HTMLElement>(".strategy-combined-name");
    const contractQty = document.querySelector<HTMLElement>(".contract-qty");
    const contractType = document.querySelector<HTMLElement>(".contract-type");
    const contractDte = document.querySelector<HTMLElement>(".contract-dte");
    const contractEquity = document.querySelector<HTMLElement>(
      ".contract-identity-equity .contract-instrument",
    );
    if (
      !table ||
      !header ||
      !symbolCell ||
      !strategyCell ||
      !legCell ||
      !contractStrip ||
      !dot ||
      !hierarchyDot ||
      !positivePnl ||
      !negativePnl ||
      !symbolLabel ||
      !singleStrategyName ||
      !combinedStrategyName ||
      !contractQty ||
      !contractType ||
      !contractDte
    ) {
      return { ok: false as const, reason: "missing-nodes" };
    }

    const cs = getComputedStyle(table);
    const tokens = {
      headerBg: cs.getPropertyValue("--pt-header-bg").trim(),
      rowBg: cs.getPropertyValue("--pt-row-bg").trim(),
      rowAlt: cs.getPropertyValue("--pt-row-alt").trim(),
      legBg: cs.getPropertyValue("--pt-leg-bg").trim(),
      segBg: cs.getPropertyValue("--pt-seg-bg").trim(),
      segRule: cs.getPropertyValue("--pt-seg-rule").trim(),
      text: cs.getPropertyValue("--pt-text").trim(),
      textIdentity: cs.getPropertyValue("--pt-text-identity").trim(),
      textSoft: cs.getPropertyValue("--pt-text-soft").trim(),
      dot: cs.getPropertyValue("--pt-dot").trim(),
      good: cs.getPropertyValue("--pt-good").trim(),
      danger: cs.getPropertyValue("--pt-danger").trim(),
      focus: cs.getPropertyValue("--pt-focus").trim(),
    };

    const headerBg = parseOklch(getComputedStyle(header).backgroundColor);
    const symbolBg = parseOklch(getComputedStyle(symbolCell).backgroundColor);
    const strategyBg = parseOklch(getComputedStyle(strategyCell).backgroundColor);
    const legBg = parseOklch(getComputedStyle(legCell).backgroundColor);
    const stripBg = parseOklch(getComputedStyle(contractStrip).backgroundColor);
    const stripBorder = parseOklch(getComputedStyle(contractStrip).borderTopColor);
    const dotBg = parseOklch(getComputedStyle(dot).backgroundColor);
    const hierarchyDotBg = parseOklch(getComputedStyle(hierarchyDot).backgroundColor);
    const posColor = parseOklch(getComputedStyle(positivePnl).color);
    const negColor = parseOklch(getComputedStyle(negativePnl).color);
    const primaryText = parseOklch(getComputedStyle(table).color);
    const symbolColor = parseOklch(getComputedStyle(symbolLabel).color);
    const singleNameColor = parseOklch(getComputedStyle(singleStrategyName).color);
    const combinedNameColor = parseOklch(getComputedStyle(combinedStrategyName).color);
    const qtyColor = parseOklch(getComputedStyle(contractQty).color);
    const typeColor = parseOklch(getComputedStyle(contractType).color);
    const dteColor = parseOklch(getComputedStyle(contractDte).color);
    const equityColor = contractEquity
      ? parseOklch(getComputedStyle(contractEquity).color)
      : null;

    const tokenRow = parseTokenOklch(tokens.rowBg);
    const tokenHeader = parseTokenOklch(tokens.headerBg);
    const tokenDot = parseTokenOklch(tokens.dot);
    const tokenGood = parseTokenOklch(tokens.good);
    const tokenDanger = parseTokenOklch(tokens.danger);
    const tokenSeg = parseTokenOklch(tokens.segBg);
    const tokenSegRule = parseTokenOklch(tokens.segRule);
    const tokenText = parseTokenOklch(tokens.text);
    const tokenIdentity = parseTokenOklch(tokens.textIdentity);
    const tokenSoft = parseTokenOklch(tokens.textSoft);

    if (
      !headerBg ||
      !symbolBg ||
      !strategyBg ||
      !legBg ||
      !stripBg ||
      !stripBorder ||
      !dotBg ||
      !hierarchyDotBg ||
      !posColor ||
      !negColor ||
      !primaryText ||
      !symbolColor ||
      !singleNameColor ||
      !combinedNameColor ||
      !qtyColor ||
      !typeColor ||
      !dteColor ||
      !tokenRow ||
      !tokenHeader ||
      !tokenDot ||
      !tokenGood ||
      !tokenDanger ||
      !tokenSeg ||
      !tokenSegRule ||
      !tokenText ||
      !tokenIdentity ||
      !tokenSoft
    ) {
      return {
        ok: false as const,
        reason: "parse-oklch",
        raw: {
          headerBg: getComputedStyle(header).backgroundColor,
          strategyBg: getComputedStyle(strategyCell).backgroundColor,
          tokens,
        },
      };
    }

    return {
      ok: true as const,
      tokens,
      headerBg,
      symbolBg,
      strategyBg,
      legBg,
      stripBg,
      stripBorder,
      dotBg,
      hierarchyDotBg,
      posColor,
      negColor,
      primaryText,
      symbolColor,
      singleNameColor,
      combinedNameColor,
      qtyColor,
      typeColor,
      dteColor,
      equityColor,
      tokenRow,
      tokenHeader,
      tokenDot,
      tokenGood,
      tokenDanger,
      tokenSeg,
      tokenSegRule,
      tokenText,
      tokenIdentity,
      tokenSoft,
    };
  });

  expect(darkPalette, JSON.stringify(darkPalette)).toMatchObject({ ok: true });
  if (darkPalette.ok) {
    // Scoped tokens present as oklch charcoal / semantic values.
    expect(darkPalette.tokens.headerBg).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.rowBg).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.legBg).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.segBg).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.text).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.textIdentity).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.textSoft).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.dot).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.good).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.danger).toMatch(/oklch\(/i);
    expect(darkPalette.tokens.focus).toBeTruthy();

    // Near-zero chroma charcoal (no olive/brown hue bias).
    for (const band of [
      darkPalette.headerBg,
      darkPalette.symbolBg,
      darkPalette.strategyBg,
      darkPalette.legBg,
      darkPalette.stripBg,
      darkPalette.tokenRow,
      darkPalette.tokenHeader,
      darkPalette.tokenSeg,
    ]) {
      expect(band.c).toBeLessThan(0.02);
      expect(band.l).toBeLessThan(40);
    }

    // Distinct hierarchy bands + lighter contract strip than strategy base.
    expect(darkPalette.legBg.l).toBeGreaterThan(darkPalette.tokenRow.l - 0.5);
    expect(darkPalette.stripBg.l).toBeGreaterThan(darkPalette.strategyBg.l);
    expect(darkPalette.stripBorder.l).toBeGreaterThan(darkPalette.stripBg.l);
    expect(darkPalette.tokenSegRule.l).toBeGreaterThan(darkPalette.tokenSeg.l);

    // Primary labels remain light and readable on charcoal.
    expect(darkPalette.primaryText.l).toBeGreaterThan(90);
    expect(darkPalette.primaryText.c).toBeLessThan(0.03);

    // Text identity hierarchy (dark): symbol primary > strategy/position identity > soft metadata.
    // Compare relative lightness only — no brittle exact RGB literals.
    const lTol = 1.5;
    expect(Math.abs(darkPalette.symbolColor.l - darkPalette.tokenText.l)).toBeLessThanOrEqual(lTol);
    expect(Math.abs(darkPalette.singleNameColor.l - darkPalette.tokenIdentity.l)).toBeLessThanOrEqual(
      lTol,
    );
    expect(
      Math.abs(darkPalette.combinedNameColor.l - darkPalette.tokenIdentity.l),
    ).toBeLessThanOrEqual(lTol);
    expect(Math.abs(darkPalette.singleNameColor.l - darkPalette.combinedNameColor.l)).toBeLessThanOrEqual(
      lTol,
    );
    expect(Math.abs(darkPalette.qtyColor.l - darkPalette.tokenIdentity.l)).toBeLessThanOrEqual(lTol);
    expect(Math.abs(darkPalette.typeColor.l - darkPalette.tokenIdentity.l)).toBeLessThanOrEqual(lTol);
    if (darkPalette.equityColor) {
      expect(Math.abs(darkPalette.equityColor.l - darkPalette.tokenIdentity.l)).toBeLessThanOrEqual(
        lTol,
      );
    }
    expect(darkPalette.tokenText.l).toBeGreaterThan(darkPalette.tokenIdentity.l + 2);
    expect(darkPalette.tokenIdentity.l).toBeGreaterThan(darkPalette.tokenSoft.l + 2);
    expect(darkPalette.symbolColor.l).toBeGreaterThan(darkPalette.singleNameColor.l + 2);
    expect(darkPalette.singleNameColor.l).toBeGreaterThan(darkPalette.dteColor.l + 2);
    // Emphasized contract segments must not jump back to bright primary.
    expect(darkPalette.qtyColor.l).toBeLessThan(darkPalette.symbolColor.l - 2);
    expect(darkPalette.typeColor.l).toBeLessThan(darkPalette.symbolColor.l - 2);
    // DTE stays at most identity-bright (typically soft).
    expect(darkPalette.dteColor.l).toBeLessThanOrEqual(darkPalette.tokenIdentity.l + lTol);

    // Magenta markers (high chroma, magenta-ish hue near 0/350–20).
    expect(darkPalette.dotBg.c).toBeGreaterThan(0.15);
    expect(darkPalette.hierarchyDotBg.c).toBeGreaterThan(0.15);
    const magentaHue = (h: number) => h <= 30 || h >= 330;
    expect(magentaHue(darkPalette.dotBg.h)).toBeTruthy();
    expect(magentaHue(darkPalette.hierarchyDotBg.h)).toBeTruthy();
    expect(darkPalette.tokenDot.c).toBeGreaterThan(0.15);

    // Positive green (~150) vs negative coral-red (~20) — visibly distinct.
    expect(darkPalette.posColor.c).toBeGreaterThan(0.08);
    expect(darkPalette.negColor.c).toBeGreaterThan(0.08);
    expect(darkPalette.posColor.h).toBeGreaterThan(120);
    expect(darkPalette.posColor.h).toBeLessThan(180);
    expect(darkPalette.negColor.h).toBeGreaterThan(0);
    expect(darkPalette.negColor.h).toBeLessThan(45);
    expect(Math.abs(darkPalette.posColor.h - darkPalette.negColor.h)).toBeGreaterThan(80);
    expect(darkPalette.tokenGood.h).toBeGreaterThan(120);
    expect(darkPalette.tokenDanger.h).toBeLessThan(45);
  }

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);

  // Light theme: positions grid stays free of axe violations (scoped tokens).
  await page.getByRole("button", { name: "Use light theme" }).click();
  await expect
    .poll(async () =>
      page.evaluate(() => document.documentElement.getAttribute("data-theme")),
    )
    .toBe("light");
  const lightAxe = await new AxeBuilder({ page })
    .include("table.positions-by-symbol")
    .analyze();
  expect(lightAxe.violations).toEqual([]);

  // Light theme text hierarchy: symbol primary < identity < soft metadata (numeric L).
  const lightIdentity = await page.evaluate(() => {
    type Oklch = { l: number; c: number; h: number };
    const parseOklch = (value: string): Oklch | null => {
      const m = value.match(/oklch\(\s*([\d.]+)\s*%?\s+([\d.]+)\s+(-?[\d.]+)/i);
      if (!m) return null;
      let l = Number(m[1]);
      if (l <= 1) l *= 100;
      return { l, c: Number(m[2]), h: Number(m[3]) };
    };
    const table = document.querySelector<HTMLElement>("table.positions-by-symbol");
    const symbolLabel = document.querySelector<HTMLElement>(".symbol-group-symbol");
    const singleName = document.querySelector<HTMLElement>("button.strategy-open-action");
    const combinedName = document.querySelector<HTMLElement>(".strategy-combined-name");
    const contractQty = document.querySelector<HTMLElement>(".contract-qty");
    const contractDte = document.querySelector<HTMLElement>(".contract-dte");
    if (!table || !symbolLabel || !singleName || !combinedName || !contractQty || !contractDte) {
      return { ok: false as const, reason: "missing-nodes" };
    }
    const cs = getComputedStyle(table);
    const tokenText = parseOklch(cs.getPropertyValue("--pt-text").trim());
    const tokenIdentity = parseOklch(cs.getPropertyValue("--pt-text-identity").trim());
    const tokenSoft = parseOklch(cs.getPropertyValue("--pt-text-soft").trim());
    const symbolColor = parseOklch(getComputedStyle(symbolLabel).color);
    const singleNameColor = parseOklch(getComputedStyle(singleName).color);
    const combinedNameColor = parseOklch(getComputedStyle(combinedName).color);
    const qtyColor = parseOklch(getComputedStyle(contractQty).color);
    const dteColor = parseOklch(getComputedStyle(contractDte).color);
    if (
      !tokenText ||
      !tokenIdentity ||
      !tokenSoft ||
      !symbolColor ||
      !singleNameColor ||
      !combinedNameColor ||
      !qtyColor ||
      !dteColor
    ) {
      return { ok: false as const, reason: "parse-oklch" };
    }
    return {
      ok: true as const,
      tokenText,
      tokenIdentity,
      tokenSoft,
      symbolColor,
      singleNameColor,
      combinedNameColor,
      qtyColor,
      dteColor,
    };
  });
  expect(lightIdentity, JSON.stringify(lightIdentity)).toMatchObject({ ok: true });
  if (lightIdentity.ok) {
    const lTol = 1.5;
    expect(Math.abs(lightIdentity.symbolColor.l - lightIdentity.tokenText.l)).toBeLessThanOrEqual(
      lTol,
    );
    expect(
      Math.abs(lightIdentity.singleNameColor.l - lightIdentity.tokenIdentity.l),
    ).toBeLessThanOrEqual(lTol);
    expect(
      Math.abs(lightIdentity.combinedNameColor.l - lightIdentity.tokenIdentity.l),
    ).toBeLessThanOrEqual(lTol);
    expect(
      Math.abs(lightIdentity.singleNameColor.l - lightIdentity.combinedNameColor.l),
    ).toBeLessThanOrEqual(lTol);
    expect(Math.abs(lightIdentity.qtyColor.l - lightIdentity.tokenIdentity.l)).toBeLessThanOrEqual(
      lTol,
    );
    // Light theme numeric L: primary (darker) < identity < soft (lighter gray).
    expect(lightIdentity.tokenText.l).toBeLessThan(lightIdentity.tokenIdentity.l - 2);
    expect(lightIdentity.tokenIdentity.l).toBeLessThan(lightIdentity.tokenSoft.l - 2);
    expect(lightIdentity.symbolColor.l).toBeLessThan(lightIdentity.singleNameColor.l - 2);
    expect(lightIdentity.singleNameColor.l).toBeLessThan(lightIdentity.dteColor.l + lTol);
    expect(lightIdentity.qtyColor.l).toBeGreaterThan(lightIdentity.symbolColor.l + 2);
  }

  // Narrow viewport: document does not overflow; table-local overflow is allowed.
  await page.setViewportSize({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => {
    const doc = document.documentElement;
    const wrap = document.querySelector(
      ".positions-workspace-main .table-wrap, .positions-pane .table-wrap",
    ) as HTMLElement | null;
    const table = wrap?.querySelector(".positions-by-symbol") as HTMLElement | null;
    return {
      docScroll: doc.scrollWidth,
      docClient: doc.clientWidth,
      wrapClient: wrap?.clientWidth ?? 0,
      wrapScroll: wrap?.scrollWidth ?? 0,
      tableMin: table
        ? Number.parseFloat(getComputedStyle(table).minWidth || "0")
        : 0,
      wrapOverflowX: wrap ? getComputedStyle(wrap).overflowX : "",
    };
  });
  expect(overflow.docScroll).toBeLessThanOrEqual(overflow.docClient + 1);
  expect(overflow.wrapOverflowX === "auto" || overflow.wrapOverflowX === "scroll").toBeTruthy();
  // Table may be wider than the wrap (local scroll); wrap itself stays within the viewport.
  expect(overflow.wrapClient).toBeLessThanOrEqual(overflow.docClient + 1);
  if (overflow.tableMin > overflow.wrapClient) {
    expect(overflow.wrapScroll).toBeGreaterThan(overflow.wrapClient);
  }
});


test("combined strategy legs always visible: no disclosure, filters, analysis, a11y", async ({
  page,
}) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  // No per-strategy disclosure controls remain.
  await expect(page.getByRole("button", { name: /Expand legs|Collapse legs/i })).toHaveCount(0);
  await expect(page.locator(".strategy-legs-toggle")).toHaveCount(0);
  await expect(page.locator(".strategy-legs-chevron")).toHaveCount(0);
  await expect(page.locator("#strategy-legs-panel-strat-qqq")).toHaveCount(0);
  await expect(page.locator("table.strategy-legs-table")).toHaveCount(0);

  const qqqStrategyRow = page
    .locator("tr[data-level='1'].combined-strategy-row")
    .filter({ hasText: "Iron Condor" });
  const aaplStrategyRow = page
    .locator("tr[data-level='1'].combined-strategy-row")
    .filter({ hasText: "Covered Call" });
  await expect(qqqStrategyRow.locator(".strategy-combined-name")).toHaveText("Iron Condor");
  await expect(qqqStrategyRow.locator(".strategy-leg-count")).toHaveText("4 legs");
  await expect(aaplStrategyRow.locator(".strategy-combined-name")).toHaveText("Covered Call");
  await expect(aaplStrategyRow.locator(".strategy-leg-count")).toHaveText("2 legs");
  // Combined name is static text — not a button with disclosure semantics.
  await expect(qqqStrategyRow.locator(".strategy-combined-name")).toHaveJSProperty(
    "tagName",
    "SPAN",
  );

  // Legs always visible under every expanded symbol.
  const qqqLegs = page.locator(
    "tr[data-level='2'][data-parent-strategy='strat-qqq']:visible",
  );
  const aaplLegs = page.locator(
    "tr[data-level='2'][data-parent-strategy='strat-aapl-cc']:visible",
  );
  await expect(qqqLegs).toHaveCount(4);
  await expect(aaplLegs).toHaveCount(2);
  await expect(qqqLegs.first()).toContainText("+1");
  await expect(qqqLegs.nth(1)).toContainText("−1");
  await expect(qqqLegs.filter({ hasText: "P" }).first()).toBeVisible();
  await expect(qqqLegs.filter({ hasText: "C" }).first()).toBeVisible();
  await expect(qqqLegs.filter({ hasText: "$470" })).toHaveCount(1);
  await expect(qqqLegs.filter({ hasText: "$475" })).toHaveCount(1);
  await expect(qqqLegs.filter({ hasText: "$505" })).toHaveCount(1);
  await expect(qqqLegs.filter({ hasText: "$510" })).toHaveCount(1);
  await expect(aaplLegs.first()).toContainText("+100");
  await expect(aaplLegs.first()).toContainText("Equity");
  await expect(aaplLegs.nth(1)).toContainText("−1");
  await expect(aaplLegs.nth(1)).toContainText("C");
  await expect(aaplLegs.filter({ hasText: "$220" })).toHaveCount(1);
  // Legs align under the main table columns (no nested thead).
  await expect(page.locator("table.positions-by-symbol thead tr th")).toHaveCount(8);
  // No account identifiers in leg rows.
  await expect(qqqLegs.first()).not.toContainText("public-account-id");
  await expect(aaplLegs.first()).not.toContainText("public-account-id");

  // Combined vs single strategy hierarchy dots/names share the same x-origin.
  await page.setViewportSize({ width: 1280, height: 800 });
  const alignment = await page.evaluate(() => {
    const combined = Array.from(
      document.querySelectorAll<HTMLElement>("tr[data-level='1'].combined-strategy-row"),
    ).find((row) => (row.textContent || "").includes("Iron Condor"));
    const single = Array.from(
      document.querySelectorAll<HTMLElement>("tr[data-level='1']"),
    ).find((row) => {
      const open = row.querySelector(".strategy-open-action");
      return (open?.textContent || "").includes("Short Put");
    });
    const cDot = combined?.querySelector<HTMLElement>(".position-hierarchy-dot");
    const cName = combined?.querySelector<HTMLElement>(".strategy-combined-name");
    const sDot = single?.querySelector<HTMLElement>(".position-hierarchy-dot");
    const sName = single?.querySelector<HTMLElement>(".strategy-open-action");
    if (!cDot || !cName || !sDot || !sName) return { ok: false as const };
    return {
      ok: true as const,
      dotDelta: Math.abs(cDot.getBoundingClientRect().left - sDot.getBoundingClientRect().left),
      nameDelta: Math.abs(
        cName.getBoundingClientRect().left - sName.getBoundingClientRect().left,
      ),
    };
  });
  expect(alignment.ok).toBeTruthy();
  if (alignment.ok) {
    expect(alignment.dotDelta).toBeLessThanOrEqual(1);
    expect(alignment.nameDelta).toBeLessThanOrEqual(1);
  }

  // Symbol collapse hides strategy + legs; reopen restores both immediately.
  const aaplSymbolToggle = page.locator("button.symbol-group-toggle").filter({ hasText: "AAPL" });
  await aaplSymbolToggle.click();
  await expect(aaplSymbolToggle).toHaveAttribute("aria-expanded", "false");
  await expect(aaplStrategyRow).toBeHidden();
  await expect(
    page.locator("tr[data-level='2'][data-parent-strategy='strat-aapl-cc']:visible"),
  ).toHaveCount(0);
  // QQQ combined strategy + legs remain visible.
  await expect(qqqStrategyRow).toBeVisible();
  await expect(qqqLegs).toHaveCount(4);
  await aaplSymbolToggle.click();
  await expect(aaplSymbolToggle).toHaveAttribute("aria-expanded", "true");
  await expect(aaplStrategyRow).toBeVisible();
  await expect(
    page.locator("tr[data-level='2'][data-parent-strategy='strat-aapl-cc']:visible"),
  ).toHaveCount(2);

  // Stock/Options filter: Options off hides combined strategies and their legs.
  const stockToggle = page.getByRole("checkbox", { name: /Show stock positions/i });
  const optionsToggle = page.getByRole("checkbox", { name: /Show options positions/i });
  await stockToggle.uncheck();
  await expect(aaplStrategyRow).toBeVisible();
  await expect(
    page.locator("tr[data-level='2'][data-parent-strategy='strat-aapl-cc']:visible"),
  ).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toHaveCount(0);

  await optionsToggle.uncheck();
  await stockToggle.check();
  await expect(
    page.locator("tr[data-level='1'].combined-strategy-row"),
  ).toHaveCount(0);
  await expect(page.locator("tr[data-level='2']:visible")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toBeVisible();
  // Re-enable Options: combined strategies and legs restore immediately (no expand step).
  await optionsToggle.check();
  await expect(aaplStrategyRow).toBeVisible();
  await expect(
    page.locator("tr[data-level='2'][data-parent-strategy='strat-aapl-cc']:visible"),
  ).toHaveCount(2);
  await expect(qqqStrategyRow).toBeVisible();
  await expect(
    page.locator("tr[data-level='2'][data-parent-strategy='strat-qqq']:visible"),
  ).toHaveCount(4);

  // Catalyst content still present on strategy rows.
  await expect(page.getByText("No confirmed catalyst found").first()).toBeVisible();

  // Analyze opens the strategy drawer.
  await page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();

  // Single-leg still opens analysis directly.
  await page.getByRole("button", { name: "Open SPY Short Put" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");

  // Roll ledger rail still present beside positions.
  await expect(page.getByRole("heading", { name: "Roll ledger" })).toBeVisible();
  await expect(page.locator(".roll-ledger-row")).toHaveCount(3);

  for (const size of [
    { width: 1280, height: 800 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(size);
    const overflow = await page.evaluate(() => {
      const doc = document.documentElement;
      return { scrollWidth: doc.scrollWidth, clientWidth: doc.clientWidth };
    });
    expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth + 1);
  }

  await page.setViewportSize({ width: 1280, height: 800 });
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("keyboard navigation reaches sections and respects reduced motion", async ({ page }) => {
  await mockDashboardApis(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();

  await page.getByRole("button", { name: "Positions" }).focus();
  await expect(page.getByRole("button", { name: "Positions" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Settings" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "Portfolio and history" })).toBeVisible();

  // Skip link is present for keyboard users.
  await expect(page.locator(".skip-link")).toHaveAttribute("href", "#main-content");

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("responsive layouts remain usable at narrow and large widths", async ({ page }) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("complementary", { name: "Primary navigation" })).toBeVisible();
  await expect(page.getByRole("status").filter({ hasText: /read-only/i })).toBeVisible();
  // Phone bottom nav: six equal columns still span the available rail width.
  const phoneNavGeometry = await page.evaluate(() => {
    const rail = document.querySelector(".navigation-rail");
    const items = Array.from(document.querySelectorAll(".nav-item"));
    if (!rail || items.length !== 6) return { ok: false as const };
    const railRect = rail.getBoundingClientRect();
    const itemRects = items.map((el) => el.getBoundingClientRect());
    const first = itemRects[0]!;
    const last = itemRects[itemRects.length - 1]!;
    return {
      ok: true as const,
      itemCount: items.length,
      leftInset: first.left - railRect.left,
      rightInset: railRect.right - last.right,
      spansRail: first.left - railRect.left <= 8 && railRect.right - last.right <= 8,
      equalWidth: itemRects.every(
        (r) => Math.abs(r.width - first.width) <= 2,
      ),
    };
  });
  expect(phoneNavGeometry.ok).toBe(true);
  if (phoneNavGeometry.ok) {
    expect(phoneNavGeometry.itemCount).toBe(6);
    expect(phoneNavGeometry.spansRail).toBe(true);
    expect(phoneNavGeometry.equalWidth).toBe(true);
  }
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  // Phone simplified layout: editing controls for watchlist are absent.
  await page.getByRole("button", { name: "Markets" }).click();
  await expect(page.getByRole("heading", { name: "Market overview" })).toBeVisible();
  await expect(page.getByLabel("Add watchlist symbol")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Add", exact: true })).toHaveCount(0);
  // Alerts remain useful for reading.
  await page.getByRole("button", { name: "Alerts" }).click();
  await expect(page.getByRole("heading", { name: "Alerts", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Acknowledge" })).toHaveCount(0);
  // Settings panel: every mutation control is hidden/disabled at phone width.
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("heading", { name: "Provider and notifications" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save recommendation settings" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Save catalyst settings" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Create backup" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Apply retention" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Save retention settings" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Restore" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Enable monitoring" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Disable monitoring" })).toHaveCount(0);
  await expect(page.getByText(/read-only on phone/i).first()).toBeVisible();
  const narrowAxe = await new AxeBuilder({ page }).analyze();
  expect(narrowAxe.violations).toEqual([]);

  // Desktop controls return at wide width.
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.getByRole("button", { name: "Overview" }).click();
  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();
  await expect(page.getByText(/Phone layout is simplified/i)).toHaveCount(0);
  await page.getByRole("button", { name: "Markets" }).click();
  await expect(page.getByLabel("Add watchlist symbol")).toBeVisible();
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("button", { name: "Save recommendation settings" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save catalyst settings" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Create backup" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Apply retention" })).toBeVisible();

  // Wide rail: tab outer edges align with brand lockup content inset (not rail edge).
  const railGeometry = await page.evaluate(() => {
    const rail = document.querySelector(".navigation-rail");
    const brand = document.querySelector(".brand-lockup");
    const brandContent = brand?.querySelector("span");
    const items = Array.from(document.querySelectorAll<HTMLElement>(".nav-item"));
    const active = document.querySelector<HTMLElement>(".nav-item.active");
    if (!rail || !brand || !brandContent || items.length !== 6 || !active) {
      return { ok: false as const, reason: "missing rail elements" };
    }

    const railRect = rail.getBoundingClientRect();
    const brandContentLeft = brandContent.getBoundingClientRect().left;
    const itemRects = items.map((el) => el.getBoundingClientRect());
    const lefts = itemRects.map((r) => r.left);
    const rights = itemRects.map((r) => r.right);
    const leftGaps = lefts.map((left) => left - railRect.left);
    const rightGaps = rights.map((right) => railRect.right - right);
    const beforeLeft = getComputedStyle(active, "::before").left;

    return {
      ok: true as const,
      brandContentLeft,
      lefts,
      rights,
      leftGaps,
      rightGaps,
      beforeLeft,
      itemCount: items.length,
    };
  });
  expect(railGeometry.ok).toBe(true);
  if (railGeometry.ok) {
    expect(railGeometry.itemCount).toBe(6);
    const sharedLeft = railGeometry.lefts[0]!;
    const sharedRight = railGeometry.rights[0]!;
    for (let i = 0; i < railGeometry.lefts.length; i += 1) {
      expect(Math.abs(railGeometry.lefts[i]! - sharedLeft)).toBeLessThanOrEqual(1);
      expect(Math.abs(railGeometry.rights[i]! - sharedRight)).toBeLessThanOrEqual(1);
      expect(Math.abs(railGeometry.lefts[i]! - railGeometry.brandContentLeft)).toBeLessThanOrEqual(1);
      expect(Math.abs(railGeometry.leftGaps[i]! - railGeometry.rightGaps[i]!)).toBeLessThanOrEqual(1);
    }
    // Active indicator flush with the inset tab box (not the rail edge).
    expect(railGeometry.beforeLeft).toBe("0px");
  }

  const wideAxe = await new AxeBuilder({ page }).analyze();
  expect(wideAxe.violations).toEqual([]);
});

test("strategy drawer backdrop stays translucent on hover (not opaque surface-3)", async ({
  page,
}) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  // Desktop layout: .drawer-backdrop is hidden below 960px.
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  async function assertBackdropTranslucentOnHover(theme: "dark" | "light") {
    if (theme === "light") {
      await page.getByRole("button", { name: "Use light theme" }).click();
      await expect
        .poll(() => page.evaluate(() => document.documentElement.getAttribute("data-theme")))
        .toBe("light");
    }

    // Combined multi-leg strategy with a detail fixture (strat-aapl-cc).
    await page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();

    const backdrop = page.locator("button.drawer-backdrop");
    await expect(backdrop).toBeVisible();

    // Real pointer hover — exercises the global button:hover cascade.
    await backdrop.hover({ position: { x: 24, y: 24 } });

    const paint = await backdrop.evaluate((el) => {
      const style = getComputedStyle(el);
      const backgroundColor = style.backgroundColor;

      const probe = document.createElement("div");
      probe.style.background = "var(--surface-3)";
      probe.style.position = "fixed";
      probe.style.left = "-9999px";
      document.body.appendChild(probe);
      const surface3 = getComputedStyle(probe).backgroundColor;
      probe.remove();

      const parseAlpha = (value: string): number | null => {
        const rgba = value.match(
          /^rgba?\(\s*[\d.]+%?\s*[,\s/]+[\d.]+%?\s*[,\s/]+[\d.]+%?(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/i,
        );
        if (rgba) {
          if (rgba[1] === undefined) return 1;
          return rgba[1].endsWith("%") ? Number.parseFloat(rgba[1]) / 100 : Number.parseFloat(rgba[1]);
        }
        const slash = value.match(/\/\s*([\d.]+%?)\s*\)/);
        if (slash) {
          return slash[1].endsWith("%")
            ? Number.parseFloat(slash[1]) / 100
            : Number.parseFloat(slash[1]);
        }
        // rgb()/oklab()/oklch()/color() without a slash channel is opaque.
        if (/^(rgb|oklab|oklch|color)\(/i.test(value)) return 1;
        return null;
      };

      return {
        backgroundColor,
        surface3,
        alpha: parseAlpha(backgroundColor),
      };
    });

    expect(paint.backgroundColor).not.toBe(paint.surface3);
    expect(paint.alpha).not.toBeNull();
    expect(paint.alpha!).toBeLessThan(1);
    expect(paint.alpha!).toBeGreaterThan(0.1);

    // Drawer and underlying Positions content remain visible through the dimmer.
    await expect(dialog).toBeVisible();
    await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }),
    ).toBeVisible();

    // Backdrop still closes the drawer.
    await backdrop.click({ position: { x: 24, y: 24 } });
    await expect(dialog).not.toBeVisible();
  }

  await assertBackdropTranslucentOnHover("dark");
  await assertBackdropTranslucentOnHover("light");
});

test("visual contract: overview structure is stable", async ({ page }) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Held-symbol catalysts" })).toBeVisible();
  await expect(page.getByText("Provider ledger")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Codex signals" })).toBeVisible();

  // Deterministic visual contract: landmark structure and primary regions.
  const landmarks = await page.evaluate(() => {
    const roles = ["banner", "navigation", "main", "complementary"];
    return roles.map((role) => ({
      role,
      count: document.querySelectorAll(`[role="${role}"], ${role === "main" ? "main" : role === "navigation" ? "nav" : role === "banner" ? "header" : "aside"}`).length,
    }));
  });
  expect(landmarks.some((item) => item.role === "main" && item.count >= 1)).toBeTruthy();
  expect(landmarks.some((item) => item.role === "complementary" && item.count >= 1)).toBeTruthy();

  // Structural snapshot of key visible headings (stable contract).
  const headings = await page.locator("h1, h2").allTextContents();
  const normalized = headings.map((text) => text.trim()).filter(Boolean).sort();
  expect(normalized).toEqual(
    expect.arrayContaining([
      "Decision field",
      "Held-symbol catalysts",
      "Codex signals",
      "Position Pilot — Overview",
    ]),
  );

  // Hierarchy: actionable recommendations appear before risk field in DOM order.
  const hierarchy = await page.evaluate(() => {
    const main = document.querySelector(".workspace-main");
    if (!main) return { ok: false };
    const order = Array.from(main.querySelectorAll("h2"))
      .map((node) => (node.textContent || "").trim())
      .filter(Boolean);
    const rec = order.indexOf("Codex signals");
    const risk = order.indexOf("Risk field");
    const providers = order.indexOf("Provider ledger");
    return {
      ok: rec >= 0 && risk >= 0 && providers >= 0 && rec < risk && risk < providers,
      order,
    };
  });
  expect(hierarchy.ok).toBeTruthy();

  // Real deterministic screenshot baseline (animations reduced; maxDiffPixelRatio in config).
  await expect(page.locator(".workspace-main")).toHaveScreenshot("overview-workspace.png", {
    animations: "disabled",
  });
});

test("content density and no page overflow at desktop, tablet, and phone", async ({ page }) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();

  async function assertDensityAndOverflow(width: number, height: number) {
    await page.setViewportSize({ width, height });
    await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();

    const metrics = await page.evaluate(() => {
      const doc = document.documentElement;
      const main = document.querySelector(".workspace-main") as HTMLElement | null;
      const header = document.querySelector(".workspace-header") as HTMLElement | null;
      const masthead = document.querySelector(".portfolio-masthead") as HTMLElement | null;
      const priority = document.querySelector(".priority-band") as HTMLElement | null;
      const sections = Array.from(document.querySelectorAll(".workspace-main .panel-section"));
      const visibleSections = sections.filter((el) => {
        const rect = el.getBoundingClientRect();
        return rect.bottom > 0 && rect.top < window.innerHeight && rect.height > 0;
      });
      return {
        scrollWidth: doc.scrollWidth,
        clientWidth: doc.clientWidth,
        headerHeight: header?.getBoundingClientRect().height ?? 0,
        mastheadHeight: masthead?.getBoundingClientRect().height ?? 0,
        priorityInView: priority
          ? priority.getBoundingClientRect().top < window.innerHeight
          : false,
        visibleSectionCount: visibleSections.length,
        bodyFontSize: Number.parseFloat(getComputedStyle(document.body).fontSize),
      };
    });

    // No horizontal page overflow (table internal scroll is fine).
    expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.clientWidth + 1);
    // Compact chrome: header and masthead should not dominate the viewport.
    expect(metrics.headerHeight).toBeLessThan(80);
    expect(metrics.mastheadHeight).toBeLessThan(160);
    // Actionable priority band should be on-screen at desktop/tablet overview.
    if (width >= 768) {
      expect(metrics.priorityInView).toBeTruthy();
      // More than a single card worth of content above the fold.
      expect(metrics.visibleSectionCount).toBeGreaterThanOrEqual(2);
    }
    // Body text: ≥16px on phone and narrow/side-panel; ≥15.5px on wide canvases.
    if (width <= 1024) {
      expect(metrics.bodyFontSize).toBeGreaterThanOrEqual(16);
    } else {
      expect(metrics.bodyFontSize).toBeGreaterThanOrEqual(15.5);
    }
  }

  await assertDensityAndOverflow(1440, 900);
  await assertDensityAndOverflow(1024, 768);
  await assertDensityAndOverflow(390, 844);

  // Coarse-pointer touch targets: assert CSS contract via CSSOM (stable; no
  // device-descriptor coupling). When the browser reports coarse pointer,
  // also sample live control heights.
  const touchContract = await page.evaluate(() => {
    const touchToken = getComputedStyle(document.documentElement)
      .getPropertyValue("--control-h-touch")
      .trim();
    let hasCoarseRule = false;
    let expandsButtons = false;
    let expandsFormControls = false;
    for (const sheet of Array.from(document.styleSheets)) {
      let rules: CSSRuleList;
      try {
        rules = sheet.cssRules;
      } catch {
        continue;
      }
      for (const rule of Array.from(rules)) {
        if (!(rule instanceof CSSMediaRule)) continue;
        const media = rule.media.mediaText.replace(/\s+/g, " ");
        if (!media.includes("pointer: coarse") && !media.includes("pointer:coarse")) {
          continue;
        }
        hasCoarseRule = true;
        for (const inner of Array.from(rule.cssRules)) {
          if (!(inner instanceof CSSStyleRule)) continue;
          const minH = inner.style.getPropertyValue("min-height");
          const usesTouch =
            minH.includes("var(--control-h-touch)") ||
            minH === touchToken ||
            minH === "2.75rem" ||
            minH === "44px";
          if (!usesTouch) continue;
          if (/\bbutton\b/.test(inner.selectorText)) expandsButtons = true;
          if (/\b(input|select|textarea)\b/.test(inner.selectorText)) {
            expandsFormControls = true;
          }
        }
      }
    }

    const coarseMatches = window.matchMedia("(pointer: coarse)").matches;
    let sampledMinHeightPx: number | null = null;
    if (coarseMatches) {
      const sample =
        (document.querySelector("button.icon-action") as HTMLElement | null) ??
        (document.querySelector("button") as HTMLElement | null);
      if (sample) {
        sampledMinHeightPx = sample.getBoundingClientRect().height;
      }
    }

    return {
      touchToken,
      hasCoarseRule,
      expandsButtons,
      expandsFormControls,
      coarseMatches,
      sampledMinHeightPx,
    };
  });
  expect(touchContract.touchToken).toMatch(/2\.75rem|44px/);
  expect(touchContract.hasCoarseRule).toBeTruthy();
  expect(touchContract.expandsButtons).toBeTruthy();
  expect(touchContract.expandsFormControls).toBeTruthy();
  if (touchContract.coarseMatches && touchContract.sampledMinHeightPx != null) {
    // 2.75rem at typical 16px root ≈ 44px; allow subpixel rounding.
    expect(touchContract.sampledMinHeightPx).toBeGreaterThanOrEqual(43);
  }

  // Positions density: compact table rows without losing strategy identity.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  await expect(page.getByText("Short Put").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Roll ledger" })).toBeVisible();
  const tableMetrics = await page.evaluate(() => {
    const row = document.querySelector(".data-table.dense tbody tr") as HTMLElement | null;
    const wrap = document.querySelector(".table-wrap") as HTMLElement | null;
    const workspace = document.querySelector(".positions-workspace") as HTMLElement | null;
    const rail = document.querySelector(".roll-ledger-rail") as HTMLElement | null;
    const wsRect = workspace?.getBoundingClientRect();
    const railRect = rail?.getBoundingClientRect();
    return {
      rowHeight: row?.getBoundingClientRect().height ?? 0,
      pageOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
      tableScrollable: wrap ? wrap.scrollWidth >= wrap.clientWidth : false,
      sideBySide:
        wsRect != null &&
        railRect != null &&
        railRect.left > (wsRect.left + wsRect.width * 0.45),
    };
  });
  expect(tableMetrics.rowHeight).toBeGreaterThan(20);
  expect(tableMetrics.rowHeight).toBeLessThan(72);
  expect(tableMetrics.pageOverflow).toBeFalsy();
  expect(tableMetrics.sideBySide).toBeTruthy();

  // Narrow/tablet: positions first, ledger second; still no document overflow.
  await page.setViewportSize({ width: 820, height: 900 });
  await expect(page.getByRole("heading", { name: "Roll ledger" })).toBeVisible();
  const narrowMetrics = await page.evaluate(() => {
    const positions = document.getElementById("positions-heading");
    const ledger = document.getElementById("roll-ledger-heading");
    const pTop = positions?.getBoundingClientRect().top ?? 0;
    const lTop = ledger?.getBoundingClientRect().top ?? 0;
    return {
      stacked: lTop > pTop + 20,
      pageOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    };
  });
  expect(narrowMetrics.stacked).toBeTruthy();
  expect(narrowMetrics.pageOverflow).toBeFalsy();
});

test("responsive typography is readable on wide canvas, side panel, phone, and desktop", async ({
  page,
}) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  await expect(page.getByText("Short Put").first()).toBeVisible();

  async function sampleTypeMetrics() {
    return page.evaluate(() => {
      const body = document.body;
      const nav = document.querySelector(".nav-item") as HTMLElement | null;
      const brand = document.querySelector(".brand-lockup") as HTMLElement | null;
      const railFoot = document.querySelector(".rail-foot") as HTMLElement | null;
      const strategy =
        (document.querySelector(".strategy-combined-name") as HTMLElement | null) ??
        (document.querySelector(".strategy-open-action") as HTMLElement | null) ??
        (document.querySelector(".symbol-group-symbol") as HTMLElement | null);
      const contract =
        (document.querySelector(".contract-identity") as HTMLElement | null) ??
        (document.querySelector(".contract-strike") as HTMLElement | null);
      // Prefer primary positions P/L tabular values (not compact meta counts).
      const tabular =
        (document.querySelector(".symbol-group-pnl.tabular") as HTMLElement | null) ??
        (document.querySelector(".pnl-metric-value.tabular") as HTMLElement | null) ??
        (document.querySelector(
          ".data-table.dense tbody td.tabular.positions-td-pnl",
        ) as HTMLElement | null) ??
        (document.querySelector(
          ".data-table.dense tbody td.tabular",
        ) as HTMLElement | null);
      const denseCell =
        (tabular?.closest("td, th") as HTMLElement | null) ??
        (document.querySelector(".data-table.dense tbody td") as HTMLElement | null);
      const pnl =
        (document.querySelector(".pnl-metric-value") as HTMLElement | null) ??
        (document.querySelector(".symbol-group-pnl") as HTMLElement | null) ??
        tabular ??
        denseCell;
      const meta =
        (document.querySelector(".roll-pnl-indicator") as HTMLElement | null) ??
        (document.querySelector(".roll-status-badge") as HTMLElement | null) ??
        (document.querySelector(".pnl-metric-percent") as HTMLElement | null) ??
        (document.querySelector(".symbol-group-meta") as HTMLElement | null) ??
        (document.querySelector("table.positions-by-symbol thead th") as HTMLElement | null);
      const catalyst =
        (document.querySelector(".catalyst-inline") as HTMLElement | null) ??
        (document.querySelector(".positions-td-catalyst") as HTMLElement | null);
      const doc = document.documentElement;
      const px = (el: Element | null) =>
        el ? Number.parseFloat(getComputedStyle(el).fontSize) || 0 : 0;
      return {
        rootFontSize: px(doc),
        bodyFontSize: px(body),
        navFontSize: px(nav),
        brandFontSize: px(brand),
        railFootFontSize: px(railFoot),
        strategyFontSize: px(strategy),
        contractFontSize: px(contract),
        denseCellFontSize: px(denseCell),
        tabularFontSize: px(tabular),
        pnlFontSize: px(pnl),
        metaFontSize: px(meta),
        catalystFontSize: px(catalyst),
        pageOverflow: doc.scrollWidth > doc.clientWidth + 1,
      };
    });
  }

  function expectReadableCore(
    metrics: Awaited<ReturnType<typeof sampleTypeMetrics>>,
    opts: { minBody: number; minNav: number },
  ) {
    expect(metrics.rootFontSize).toBeGreaterThanOrEqual(17.5);
    expect(metrics.bodyFontSize).toBeGreaterThanOrEqual(opts.minBody);
    expect(metrics.navFontSize).toBeGreaterThanOrEqual(opts.minNav);
    expect(metrics.strategyFontSize).toBeGreaterThanOrEqual(13);
    expect(metrics.contractFontSize).toBeGreaterThanOrEqual(12);
    expect(metrics.pnlFontSize).toBeGreaterThanOrEqual(13);
    expect(metrics.metaFontSize).toBeGreaterThanOrEqual(11);
    // Tabular P/L must not be materially smaller than surrounding cell text.
    if (metrics.tabularFontSize > 0 && metrics.denseCellFontSize > 0) {
      expect(metrics.tabularFontSize).toBeGreaterThanOrEqual(metrics.denseCellFontSize - 0.5);
    }
    if (metrics.catalystFontSize > 0) {
      expect(metrics.catalystFontSize).toBeGreaterThanOrEqual(12);
    }
    expect(metrics.pageOverflow).toBeFalsy();
  }

  // Wide internal canvas (Codex embedded browser ~1919 CSS px): base type must
  // be large without relying on max-width: 70rem media queries.
  await page.setViewportSize({ width: 1919, height: 1080 });
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  const wideCanvas = await sampleTypeMetrics();
  expectReadableCore(wideCanvas, { minBody: 15.5, minNav: 11 });
  expect(wideCanvas.brandFontSize).toBeGreaterThanOrEqual(11);
  expect(wideCanvas.railFootFontSize).toBeGreaterThanOrEqual(11);

  // Codex side panel / tablet (~800–900 CSS px): stronger body than wide base.
  await page.setViewportSize({ width: 850, height: 900 });
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  const sidePanel = await sampleTypeMetrics();
  expectReadableCore(sidePanel, { minBody: 16, minNav: 11 });
  expect(sidePanel.bodyFontSize).toBeGreaterThanOrEqual(wideCanvas.bodyFontSize);

  // Phone: body remains comfortably large; bottom nav labels stay usable with icons.
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  const phone = await sampleTypeMetrics();
  expectReadableCore(phone, { minBody: 16, minNav: 9 });

  // Ordinary desktop width: same global 18px root / readable body baseline.
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
  const desktop = await sampleTypeMetrics();
  expectReadableCore(desktop, { minBody: 15.5, minNav: 11 });
  expect(desktop.brandFontSize).toBeGreaterThanOrEqual(11);
  expect(desktop.railFootFontSize).toBeGreaterThanOrEqual(11);
  expect(desktop.rootFontSize).toBeGreaterThanOrEqual(wideCanvas.rootFontSize - 0.5);
  expect(desktop.rootFontSize).toBeLessThanOrEqual(wideCanvas.rootFontSize + 0.5);
});

test("positions roll ledger aggregates all displayable accounts and clears on scope change", async ({
  page,
}) => {
  const rollRequests: string[] = [];
  await mockDashboardApis(page);

  // Multi-account portfolio for all-scope aggregation.
  await page.route("**/api/v1/portfolio**", (route) => {
    // Do not intercept risk; leave that to the shared mock.
    if (route.request().url().includes("/portfolio/risk")) {
      return route.fallback();
    }
    const accountId =
      new URL(route.request().url()).searchParams.get("account_id") ?? "all";
    const accounts = [
      {
        account_id: "public-account-id",
        label: "Individual 1",
        account_type: "Individual",
        net_liquidating_value: 25000,
        cash_balance: 5000,
        buying_power: 10000,
        pnl_today: 0,
        positions: [{ symbol: "SPY" }],
      },
      {
        account_id: "second-account-id",
        label: "Individual 2",
        account_type: "Individual",
        net_liquidating_value: 12000,
        cash_balance: 2000,
        buying_power: 4000,
        pnl_today: 0,
        positions: [{ symbol: "IWM" }],
      },
      // Nearly-empty inactive shell must not be fetched for the roll rail.
      {
        account_id: "inactive-shell",
        label: "Shell",
        account_type: "Individual",
        net_liquidating_value: 5,
        cash_balance: 5,
        buying_power: 0,
        pnl_today: 0,
        positions: [],
      },
    ];
    const strategies =
      accountId === "second-account-id"
        ? [
            {
              strategy_id: "strat-iwm",
              account_id: "second-account-id",
              underlying: "IWM",
              strategy_type: "Short Put",
              expiration_date: "2026-08-21",
              days_to_expiration: 21,
              quantity: 1,
              strikes: "$200",
              unrealized_pnl: 10,
              unrealized_pnl_percent: 5,
              total_delta: -8,
              total_theta: 2,
              horizon: "tactical",
              legs: [],
            },
          ]
        : accountId === "public-account-id"
          ? [
              {
                strategy_id: "strat-browser",
                account_id: "public-account-id",
                underlying: "SPY",
                strategy_type: "Short Put",
                expiration_date: "2026-08-21",
                days_to_expiration: 21,
                quantity: 1,
                strikes: "$500",
                unrealized_pnl: 40,
                unrealized_pnl_percent: 10,
                total_delta: -20,
                total_theta: 4,
                horizon: "tactical",
                legs: [],
              },
            ]
          : [
              {
                strategy_id: "strat-browser",
                account_id: "public-account-id",
                underlying: "SPY",
                strategy_type: "Short Put",
                expiration_date: "2026-08-21",
                days_to_expiration: 21,
                quantity: 1,
                strikes: "$500",
                unrealized_pnl: 40,
                unrealized_pnl_percent: 10,
                total_delta: -20,
                total_theta: 4,
                horizon: "tactical",
                legs: [],
              },
              {
                strategy_id: "strat-iwm",
                account_id: "second-account-id",
                underlying: "IWM",
                strategy_type: "Short Put",
                expiration_date: "2026-08-21",
                days_to_expiration: 21,
                quantity: 1,
                strikes: "$200",
                unrealized_pnl: 10,
                unrealized_pnl_percent: 5,
                total_delta: -8,
                total_theta: 2,
                horizon: "tactical",
                legs: [],
              },
            ];
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        schema_version: 1,
        snapshot_id: `snapshot-${accountId}`,
        captured_at: "2026-07-11T16:30:00Z",
        state: "live",
        freshness: {
          as_of: "2026-07-11T16:30:00Z",
          provider: "tastytrade",
          state: "fresh",
        },
        accounts:
          accountId === "all"
            ? accounts
            : accounts.filter((account) => account.account_id === accountId),
        strategies,
        totals: {
          net_liquidating_value: accountId === "all" ? 37000 : 25000,
          cash_balance: 7000,
          buying_power: 14000,
          unrealized_pnl: 50,
        },
        selected_account_id: accountId,
        notice: null,
      }),
    });
  });

  await page.route("**/api/v1/accounts/*/rolls", async (route) => {
    rollRequests.push(route.request().url());
    const url = route.request().url();
    if (url.includes("inactive-shell")) {
      return route.fulfill({ contentType: "application/json", body: "[]" });
    }
    if (url.includes("second-account-id")) {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify([
          {
            chain_id: "chain-second",
            account_id: "second-account-id",
            underlying: "IWM",
            strategy_type: "Short Put",
            original_open_credit: 1.0,
            chain_total_credit: 1.4,
            rolls: [
              {
                roll_id: "roll-second",
                timestamp: "2026-07-11T12:00:00Z",
                underlying: "IWM",
                strategy_type: "Short Put",
                old_strike: 200,
                new_strike: 198,
                old_dte: 7,
                new_dte: 21,
                roll_pnl: 25,
                premium_effect: 0.4,
              },
            ],
          },
        ]),
      });
    }
    if (url.includes("public-account-id")) {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify([
          {
            chain_id: "chain-public",
            account_id: "public-account-id",
            underlying: "SPY",
            strategy_type: "Short Put",
            original_open_credit: 2.5,
            chain_total_credit: 2.8,
            rolls: [
              {
                roll_id: "roll-public",
                timestamp: "2026-07-09T16:30:00Z",
                old_strike: 495,
                new_strike: 490,
                old_dte: 14,
                new_dte: 35,
                roll_pnl: 50,
                premium_effect: 0.3,
              },
            ],
          },
        ]),
      });
    }
    return route.fulfill({ contentType: "application/json", body: "[]" });
  });

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();
  // Wait until portfolio is ready so the account scope control is enabled.
  await expect(page.getByText("$37,000.00").or(page.getByText("$25,000.00")).first()).toBeVisible();
  const accountScope = page.getByRole("combobox", { name: "Account scope" });
  await expect(accountScope).toBeEnabled();

  // Switch to all accounts and open Positions — should call rolls for both displayable ids.
  await accountScope.selectOption("all");
  await expect(page.getByText("$37,000.00").first()).toBeVisible();
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Roll ledger" })).toBeVisible();
  await expect(page.locator(".roll-ledger-row")).toHaveCount(2, { timeout: 10_000 });
  await expect(page.getByText("IWM").first()).toBeVisible();
  await expect(page.getByText("SPY").first()).toBeVisible();
  // Newest first: IWM (Jul 11) before SPY (Jul 9)
  await expect(page.locator(".roll-ledger-row").nth(0)).toContainText("IWM");
  await expect(page.locator(".roll-ledger-row").nth(1)).toContainText("SPY");
  // Net: 0.3 + 0.4 = 0.70
  await expect(page.getByText("+$0.70").first()).toBeVisible();

  const allScopeRollUrls = rollRequests.filter(
    (url) => url.includes("/rolls") && !url.includes("patterns") && !url.includes("heatmap"),
  );
  expect(allScopeRollUrls.some((url) => url.includes("public-account-id"))).toBeTruthy();
  expect(allScopeRollUrls.some((url) => url.includes("second-account-id"))).toBeTruthy();
  expect(allScopeRollUrls.some((url) => url.includes("inactive-shell"))).toBeFalsy();

  // Single-account scope loads only that account; ledger must not keep the other account's events.
  rollRequests.length = 0;
  await accountScope.selectOption("public-account-id");
  await expect(page.locator(".roll-ledger-row")).toHaveCount(1, { timeout: 10_000 });
  await expect(page.locator(".roll-ledger-row").nth(0)).toContainText("SPY");
  await expect(page.getByText("IWM")).toHaveCount(0);
  const singleUrls = rollRequests.filter(
    (url) => url.includes("/rolls") && !url.includes("patterns") && !url.includes("heatmap"),
  );
  expect(singleUrls.every((url) => url.includes("public-account-id"))).toBeTruthy();
  expect(singleUrls.some((url) => url.includes("second-account-id"))).toBeFalsy();
});

function catalystEventFixture(overrides: Record<string, unknown> = {}) {
  return {
    catalyst_id: "evt-1",
    symbol: "SPY",
    headline: "Quiet headline",
    summary: "Supporting-only note",
    taxonomy: "company",
    confidence: "supporting",
    attribution: "company",
    evidence_kind: "news",
    event_at: "2026-07-11T14:00:00Z",
    sources: [{ source_id: "s1", name: "Wire", tier: "1", url: "https://example.com", provider: "test", published_at: "2026-07-11T14:00:00Z", excerpt: null }],
    rank_score: 1,
    high_impact: false,
    ...overrides,
  };
}

function symbolCatalystFixture(overrides: Record<string, unknown> = {}) {
  return {
    symbol: "SPY",
    confidence: "no_confirmed_catalyst_found",
    attribution: "none",
    summary: "No confirmed catalyst found",
    catalysts: [],
    option_mechanisms: [],
    social_side_notes: [],
    move_percent: 0.2,
    prior_close: 500,
    last_price: 501,
    meaningful_move: false,
    promoted: false,
    coverage: "complete",
    coverage_notes: [],
    quiet: true,
    freshness: {
      as_of: "2026-07-11T16:30:00Z",
      provider: "catalyst-service",
      state: "fresh",
    },
    ...overrides,
  };
}

test("catalyst risk gauges render accessible labels and semantic classes on Overview and position detail", async ({
  page,
}) => {
  await mockDashboardApis(page);

  const highSymbol = symbolCatalystFixture({
    symbol: "SPY",
    confidence: "confirmed",
    attribution: "company",
    summary: "FOMC-driven gap risk",
    meaningful_move: true,
    promoted: true,
    quiet: false,
    move_percent: 3.2,
    catalysts: [
      catalystEventFixture({
        catalyst_id: "evt-high",
        symbol: "SPY",
        headline: "Fed emergency statement",
        summary: "High-impact scheduled risk",
        confidence: "confirmed",
        high_impact: true,
      }),
      catalystEventFixture({
        catalyst_id: "evt-med",
        symbol: "SPY",
        headline: "Peer guidance cut",
        summary: "Likely secondary catalyst",
        confidence: "likely",
        high_impact: false,
      }),
      catalystEventFixture({
        catalyst_id: "evt-low",
        symbol: "SPY",
        headline: "Blog speculation",
        summary: "Supporting only",
        confidence: "supporting",
        high_impact: false,
      }),
    ],
  });

  const mediumSymbol = symbolCatalystFixture({
    symbol: "QQQ",
    confidence: "likely",
    attribution: "macro",
    summary: "Likely macro-driven move",
    meaningful_move: false,
    promoted: false,
    quiet: false,
    move_percent: 1.1,
    catalysts: [
      catalystEventFixture({
        catalyst_id: "evt-qqq",
        symbol: "QQQ",
        headline: "CPI preview chatter",
        confidence: "likely",
        high_impact: false,
      }),
    ],
  });

  const lowSymbol = symbolCatalystFixture({
    symbol: "IWM",
    confidence: "no_confirmed_catalyst_found",
    summary: "Quiet tape",
    quiet: true,
    coverage: "offline",
    cached: true,
    freshness: {
      as_of: "2026-07-11T12:00:00Z",
      provider: "catalyst-service",
      state: "stale",
    },
    catalysts: [],
  });

  await page.route("**/api/v1/catalysts**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        captured_at: "2026-07-11T16:30:00Z",
        results: [highSymbol, mediumSymbol, lowSymbol],
        settings: {
          stock_move_threshold_pct: 2,
          etf_move_threshold_pct: 1,
          news_cadence_seconds: 300,
          benzinga: { enabled: false, status: "disabled" },
          scheduled_window_hours: 72,
        },
        coverage: "incomplete",
        coverage_notes: ["Benzinga offline"],
        freshness: {
          as_of: "2026-07-11T16:30:00Z",
          provider: "catalyst-service",
          state: "stale",
        },
      }),
    }),
  );

  await page.route("**/api/v1/strategies/strat-browser", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        strategy: {
          strategy_id: "strat-browser",
          account_id: "public-account-id",
          underlying: "SPY",
          strategy_type: "Short Put",
          expiration_date: "2026-08-21",
          days_to_expiration: 21,
          quantity: 1,
          strikes: "$500",
          unrealized_pnl: 40,
          unrealized_pnl_percent: 10,
          total_delta: -20,
          total_theta: 4,
          horizon: "tactical",
          legs: [],
        },
        risk: {
          max_profit: 250,
          max_loss: 49750,
          breakevens: [497.5],
          distance_to_nearest_strike: 0,
          underlying_price: 500,
          current_pnl: 40,
          defined_risk: false,
          valuation_basis: "current_mark",
          combined: {
            delta: -20,
            gamma: 0.1,
            theta: 4,
            vega: -8,
            average_iv: 0.2,
            nearest_dte: 21,
          },
          stress: [],
        },
        market: {
          symbol: "SPY",
          price: 500,
          bid: 499.9,
          ask: 500.1,
          iv: 0.2,
          iv_rank: 35,
          iv_percentile: 40,
          liquidity_rating: 4,
          iv_environment: "normal",
          spread_percent: 0.04,
        },
        chart: {
          symbol: "SPY",
          bars: [],
          source: "massive-stocks",
          notice: null,
          prior_close: 498,
          include_extended_hours: true,
          event_markers: [],
        },
        catalyst: highSymbol,
        thesis: null,
        trade_plan: null,
        audit: [],
        rolls: [],
        events: [],
      }),
    }),
  );

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("heading", { name: "Held-symbol catalysts" })).toBeVisible();

  // Coverage remains visible and separate from risk labels.
  await expect(page.getByText(/Coverage incomplete/i)).toBeVisible();

  const overviewGauges = page.locator(".catalyst-list .catalyst-risk-gauge");
  await expect(overviewGauges).toHaveCount(3);

  const highGauge = overviewGauges.nth(0);
  await expect(highGauge).toHaveAttribute("aria-label", "Catalyst risk: High");
  await expect(highGauge).toHaveAttribute("data-risk-level", "high");
  await expect(highGauge).toHaveAttribute("data-risk-token", "danger");
  await expect(highGauge).toHaveAttribute("data-risk-segments", "3");
  await expect(highGauge).toHaveClass(/catalyst-risk-token-danger/);
  await expect(highGauge.locator(".catalyst-risk-text")).toHaveText("High");
  await expect(highGauge.locator(".catalyst-risk-seg.is-active")).toHaveCount(3);

  const mediumGauge = overviewGauges.nth(1);
  await expect(mediumGauge).toHaveAttribute("aria-label", "Catalyst risk: Medium");
  await expect(mediumGauge).toHaveAttribute("data-risk-level", "medium");
  await expect(mediumGauge).toHaveAttribute("data-risk-token", "warn");
  await expect(mediumGauge).toHaveAttribute("data-risk-segments", "2");
  await expect(mediumGauge).toHaveClass(/catalyst-risk-token-warn/);
  await expect(mediumGauge.locator(".catalyst-risk-text")).toHaveText("Medium");
  await expect(mediumGauge.locator(".catalyst-risk-seg.is-active")).toHaveCount(2);

  const lowGauge = overviewGauges.nth(2);
  await expect(lowGauge).toHaveAttribute("aria-label", "Catalyst risk: Low");
  await expect(lowGauge).toHaveAttribute("data-risk-level", "low");
  await expect(lowGauge).toHaveAttribute("data-risk-token", "good");
  await expect(lowGauge).toHaveAttribute("data-risk-segments", "1");
  await expect(lowGauge).toHaveClass(/catalyst-risk-token-good/);
  await expect(lowGauge.locator(".catalyst-risk-text")).toHaveText("Low");
  await expect(lowGauge.locator(".catalyst-risk-seg.is-active")).toHaveCount(1);

  // Stale/offline fixture still shows Low risk (coverage does not rewrite risk).
  await expect(page.getByText("Quiet tape")).toBeVisible();
  await expect(page.locator(".catalyst-list li").filter({ hasText: "IWM" }).getByText("Stale cache")).toBeVisible();

  // Confidence remains separately labeled beside risk.
  await expect(page.locator(".pill.confidence-confirmed").first()).toBeVisible();
  await expect(page.getByRole("img", { name: "Catalyst risk: High" }).first()).toBeVisible();

  // Position detail: aggregate + event-level gauges.
  await page.getByRole("button", { name: "Positions" }).click();
  await page.getByRole("button", { name: "Open SPY Short Put" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Catalysts" })).toBeVisible();

  const detailSection = page.getByRole("dialog").locator("section").filter({
    has: page.getByRole("heading", { name: "Catalysts" }),
  });
  const detailAggregate = detailSection.locator(".catalyst-row-head .catalyst-risk-gauge").first();
  await expect(detailAggregate).toHaveAttribute("aria-label", "Catalyst risk: High");
  await expect(detailAggregate).toHaveClass(/catalyst-risk-token-danger/);
  await expect(detailAggregate.locator(".catalyst-risk-seg.is-active")).toHaveCount(3);

  const eventGauges = detailSection.locator(".catalyst-event-head .catalyst-risk-gauge");
  await expect(eventGauges).toHaveCount(3);
  await expect(eventGauges.nth(0)).toHaveAttribute("aria-label", "Catalyst risk: High");
  await expect(eventGauges.nth(0)).toHaveAttribute("data-risk-token", "danger");
  await expect(eventGauges.nth(0).locator(".catalyst-risk-seg.is-active")).toHaveCount(3);
  await expect(eventGauges.nth(1)).toHaveAttribute("aria-label", "Catalyst risk: Medium");
  await expect(eventGauges.nth(1)).toHaveAttribute("data-risk-token", "warn");
  await expect(eventGauges.nth(1).locator(".catalyst-risk-seg.is-active")).toHaveCount(2);
  await expect(eventGauges.nth(2)).toHaveAttribute("aria-label", "Catalyst risk: Low");
  await expect(eventGauges.nth(2)).toHaveAttribute("data-risk-token", "good");
  await expect(eventGauges.nth(2).locator(".catalyst-risk-seg.is-active")).toHaveCount(1);

  // Confidence pill still present and distinct from risk gauge.
  await expect(detailSection.locator(".pill.confidence-confirmed")).toBeVisible();
});

test("positions show roll-adjusted P/L Open with roll badge on option contract identity", async ({
  page,
}) => {
  await mockDashboardApis(page);
  // Override portfolio after the shared mock; leave /portfolio/risk to shared handler.
  await page.route("**/api/v1/portfolio**", async (route) => {
    if (route.request().url().includes("/portfolio/risk")) {
      return route.fallback();
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        schema_version: 1,
        snapshot_id: "snap-roll-adjusted",
        captured_at: "2026-07-13T16:30:00Z",
        state: "live",
        freshness: {
          as_of: "2026-07-13T16:30:00Z",
          provider: "tastytrade",
          state: "fresh",
        },
        accounts: [
          {
            account_id: "public-account-id",
            label: "Individual 1",
            account_type: "Individual",
            net_liquidating_value: 25000,
            cash_balance: 5000,
            buying_power: 10000,
            pnl_today: 0,
            positions: [],
          },
        ],
        strategies: [
          {
            strategy_id: "mu-strangle",
            account_id: "public-account-id",
            underlying: "MU",
            strategy_type: "Short Strangle",
            expiration_date: "2025-02-21",
            days_to_expiration: 30,
            quantity: 1,
            strikes: "$800/$1400",
            unrealized_pnl: 584,
            unrealized_pnl_percent: 12,
            pnl_open: 945,
            pnl_open_percent: 18,
            roll_adjustment: 361,
            roll_count: 1,
            total_delta: -0.1,
            total_theta: 20,
            horizon: "tactical",
            legs: [
              {
                symbol: "MU 250221P00800000",
                underlying_symbol: "MU",
                quantity: 1,
                quantity_direction: "Short",
                position_type: "Equity Option",
                strike_price: 800,
                option_type: "P",
                expiration_date: "2025-02-21",
                days_to_expiration: 30,
                mark_price: 4.62,
                market_value: -462,
                unrealized_pnl: -212,
                unrealized_pnl_percent: null,
                pnl_open: -212,
                roll_adjustment: 0,
                roll_count: 0,
                roll_history_status: "none",
                delta: -0.2,
                gamma: 0.01,
                theta: 10,
                vega: -0.1,
                implied_volatility: 0.4,
                multiplier: 100,
                horizon: "tactical",
              },
              {
                symbol: "MU 250221C01400000",
                underlying_symbol: "MU",
                quantity: 1,
                quantity_direction: "Short",
                position_type: "Equity Option",
                strike_price: 1400,
                option_type: "C",
                expiration_date: "2025-02-21",
                days_to_expiration: 30,
                mark_price: 8.05,
                market_value: -805,
                unrealized_pnl: 796,
                unrealized_pnl_percent: null,
                pnl_open: 1157,
                pnl_open_percent: 25,
                roll_adjustment: 361,
                roll_count: 1,
                roll_history_status: "complete",
                roll_chain_id: "call-chain",
                delta: 0.1,
                gamma: 0.01,
                theta: 10,
                vega: -0.1,
                implied_volatility: 0.4,
                multiplier: 100,
                horizon: "tactical",
              },
            ],
          },
        ],
        totals: {
          net_liquidating_value: 25000,
          cash_balance: 5000,
          buying_power: 10000,
          unrealized_pnl: 584,
        },
        selected_account_id: "public-account-id",
        notice: null,
      }),
    });
  });

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("button", { name: "Positions" })).toBeVisible();
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  const muHeader = page.locator("tr[data-level='0']").filter({ hasText: "MU" });
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$945");

  // Combined strategy row: roll-adjusted P/L values stay, but never a roll badge.
  const strangleRow = page
    .locator("tr[data-level='1'].combined-strategy-row")
    .filter({ hasText: "Short Strangle" });
  await expect(strangleRow.locator(".pnl-metric")).toContainText("$945.00");
  await expect(strangleRow.locator(".roll-pnl-indicator")).toHaveCount(0);

  // Rolled 1400C child: exactly one R1 immediately after the contract strip.
  const callLeg = page.locator("tr[data-level='2']").filter({ hasText: "1400" });
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,157.00");
  await expect(callLeg.locator(".positions-td-pnl .roll-pnl-indicator")).toHaveCount(0);
  await expect(callLeg.locator(".roll-pnl-indicator")).toHaveCount(1);
  await expect(callLeg.locator(".roll-pnl-indicator")).toContainText("R1");
  await expect(callLeg.locator(".roll-pnl-indicator")).toHaveAttribute(
    "aria-label",
    /realized carry/,
  );
  // Badge is a sibling after .contract-identity (not nested inside the fixed strip).
  await expect(
    callLeg.locator(".leg-contract-identity-row > .contract-identity + .roll-pnl-indicator"),
  ).toHaveCount(1);
  await expect(callLeg.locator(".contract-identity .roll-pnl-indicator")).toHaveCount(0);

  // Unrolled 800P child: no roll badge.
  const putLeg = page.locator("tr[data-level='2']").filter({ hasText: "800" });
  await expect(putLeg.locator(".roll-pnl-indicator")).toHaveCount(0);

  // Hierarchy hooks remain intact with roll-adjusted display.
  await expect(page.locator("tr[data-level='0']")).toHaveCount(1);
  await expect(page.locator("tr[data-level='1']")).toHaveCount(1);
  await expect(page.locator("tr[data-level='2']:visible")).toHaveCount(2);
});


test("Roll chains table shows realized carry, lifetime credit, and status badges", async ({
  page,
}) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Roll analytics" }).click();
  await expect(page.getByRole("heading", { name: "Roll chains" })).toBeVisible();

  const table = page.getByRole("region", { name: "Roll chains table" });
  await expect(table.getByRole("columnheader", { name: "Realized carry" })).toBeVisible();
  await expect(table.getByRole("columnheader", { name: "Lifetime credit" })).toBeVisible();
  await expect(table.getByRole("columnheader", { name: "Status" })).toBeVisible();

  const firstRow = table.locator("tbody tr").first();
  await expect(firstRow.locator(".roll-status-badge")).toBeVisible();
  await expect(firstRow.locator(".roll-status-badge")).toHaveAttribute(
    "aria-label",
    /Complete|Partial/,
  );
});

test("Positions column header is P/L Open", async ({ page }) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(
    page.locator("th.positions-th-pnl"),
  ).toHaveText(/P\/L Open/);
});

/**
 * Live Positions P/L from DXLink SSE Quote — child, combined, and symbol-group
 * rows update together; unrolled sibling stays fixed; no portfolio REST refresh.
 * Uses a controllable EventSource so the stream stays open (production onerror
 * remains strict and is not weakened for short mock bodies).
 */
test("live market Quote SSE updates Positions P/L overlay without REST refresh", async ({
  page,
}) => {
  await installControllableEventSource(page);
  await mockDashboardApis(page);

  let portfolioHits = 0;
  await page.route("**/api/v1/portfolio**", async (route) => {
    if (route.request().url().includes("/portfolio/risk")) {
      return route.fallback();
    }
    portfolioHits += 1;
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(muLivePortfolioBody("snap-live-pnl")),
    });
  });

  await page.route("**/api/v1/streaming/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        market: { state: "live", error: null },
        account: { state: "live", error: null },
      }),
    }),
  );

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.getByRole("button", { name: "Positions" })).toBeVisible();
  // Wait for live stream status (onopen → status recheck).
  await expect(page.locator(".local-clock.live")).toBeVisible({ timeout: 10_000 });
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  // Snapshot baseline before tick.
  const callLeg = page.locator("tr[data-level='2']").filter({ hasText: "1400" });
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,557.00");

  await sseEmit(page, LIVE_CALL_QUOTE);

  // Live mark 3.725 → short raw 1601-372.5=1228.5 + carry 361 = 1589.5
  // Combined: -212 + 1589.5 = 1377.5; put sibling stays -212.
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,589.50", {
    timeout: 10_000,
  });
  const putLeg = page.locator("tr[data-level='2']").filter({ hasText: "800" });
  await expect(putLeg.locator(".pnl-metric")).toContainText("-$212.00");

  const strangleRow = page
    .locator("tr[data-level='1'].combined-strategy-row")
    .filter({ hasText: "Short Strangle" });
  await expect(strangleRow.locator(".pnl-metric")).toContainText("$1,377.50");

  const muHeader = page.locator("tr[data-level='0']").filter({ hasText: "MU" });
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$1,377.50");

  // After live overlay settles, further SSE ticks must not trigger portfolio REST.
  const hitsAfterLive = portfolioHits;
  await sseEmit(page, LIVE_CALL_QUOTE);
  await page.waitForTimeout(400);
  expect(portfolioHits).toBe(hitsAfterLive);
});

test("SSE transport error immediately restores snapshot Positions P/L", async ({
  page,
}) => {
  await installControllableEventSource(page);
  await mockDashboardApis(page);

  await page.route("**/api/v1/portfolio**", async (route) => {
    if (route.request().url().includes("/portfolio/risk")) {
      return route.fallback();
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(muLivePortfolioBody("snap-sse-error")),
    });
  });

  // Backend market stream still reports live — transport failure must not wait
  // on this and must still disable the overlay.
  await page.route("**/api/v1/streaming/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        market: { state: "live", error: null },
        account: { state: "live", error: null },
      }),
    }),
  );

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.locator(".local-clock.live")).toBeVisible({ timeout: 10_000 });
  await page.getByRole("button", { name: "Positions" }).click();

  const callLeg = page.locator("tr[data-level='2']").filter({ hasText: "1400" });
  const putLeg = page.locator("tr[data-level='2']").filter({ hasText: "800" });
  const strangleRow = page
    .locator("tr[data-level='1'].combined-strategy-row")
    .filter({ hasText: "Short Strangle" });
  const muHeader = page.locator("tr[data-level='0']").filter({ hasText: "MU" });

  await sseEmit(page, LIVE_CALL_QUOTE);
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,589.50", {
    timeout: 10_000,
  });
  await expect(strangleRow.locator(".pnl-metric")).toContainText("$1,377.50");
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$1,377.50");

  // Simulate browser SSE pipe failure while hub status remains "live".
  await sseError(page);

  await expect(page.locator(".local-clock.degraded")).toBeVisible({ timeout: 5_000 });
  // Authoritative snapshot values restored for child, parent, and symbol total.
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,557.00");
  await expect(putLeg.locator(".pnl-metric")).toContainText("-$212.00");
  await expect(strangleRow.locator(".pnl-metric")).toContainText("$1,345.00");
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$1,345");

  // Reopen + status live without a new quote must still show snapshot (marks cleared).
  await sseReopen(page);
  await expect(page.locator(".local-clock.live")).toBeVisible({ timeout: 5_000 });
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,557.00");
  await expect(strangleRow.locator(".pnl-metric")).toContainText("$1,345.00");
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$1,345");
  await expect(callLeg.locator(".pnl-metric")).not.toContainText("$1,589.50");

  // A *new* quote repopulates marks and restores the live overlay.
  await sseEmit(page, LIVE_CALL_QUOTE);
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,589.50", {
    timeout: 10_000,
  });
  await expect(strangleRow.locator(".pnl-metric")).toContainText("$1,377.50");
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$1,377.50");
});

test("degraded streaming keeps snapshot Positions P/L despite market Quote SSE", async ({
  page,
}) => {
  await installControllableEventSource(page);
  await mockDashboardApis(page);

  await page.route("**/api/v1/portfolio**", async (route) => {
    if (route.request().url().includes("/portfolio/risk")) {
      return route.fallback();
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(muLivePortfolioBody("snap-live-degraded")),
    });
  });

  await page.route("**/api/v1/streaming/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        market: { state: "degraded", error: "stale" },
        account: { state: "degraded", error: "stale" },
      }),
    }),
  );

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  await sseEmit(page, LIVE_CALL_QUOTE);
  await page.waitForTimeout(300);

  // Snapshot values remain — live tick must not blank or override when degraded.
  const callLeg = page.locator("tr[data-level='2']").filter({ hasText: "1400" });
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,557.00");
  const putLeg = page.locator("tr[data-level='2']").filter({ hasText: "800" });
  await expect(putLeg.locator(".pnl-metric")).toContainText("-$212.00");
  const muHeader = page.locator("tr[data-level='0']").filter({ hasText: "MU" });
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$1,345");
});

/**
 * A newer REST snapshot (changed snapshot_id) must retire prior live marks so
 * authoritative snapshot B is shown rather than a mark received under snapshot A.
 */
test("new REST snapshot_id retires prior live marks", async ({ page }) => {
  await installControllableEventSource(page);
  await mockDashboardApis(page);

  let snapshotId = "snap-A";
  await page.route("**/api/v1/portfolio**", async (route) => {
    if (route.request().url().includes("/portfolio/risk")) {
      return route.fallback();
    }
    const body =
      snapshotId === "snap-A"
        ? muLivePortfolioBody("snap-A")
        : muLivePortfolioBody("snap-B", {
            // Distinct authoritative mark/P&L that old live overlay would contradict.
            callMark: 2.0,
            callRaw: 1401,
            callPnlOpen: 1762,
            strategyPnlOpen: 1550, // -212 + 1762
          });
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });

  await page.route("**/api/v1/streaming/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        market: { state: "live", error: null },
        account: { state: "live", error: null },
      }),
    }),
  );

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await expect(page.locator(".local-clock.live")).toBeVisible({ timeout: 10_000 });
  await page.getByRole("button", { name: "Positions" }).click();

  const callLeg = page.locator("tr[data-level='2']").filter({ hasText: "1400" });
  const strangleRow = page
    .locator("tr[data-level='1'].combined-strategy-row")
    .filter({ hasText: "Short Strangle" });
  const muHeader = page.locator("tr[data-level='0']").filter({ hasText: "MU" });

  // Mark under snapshot A → live overlay 1589.50.
  await sseEmit(page, LIVE_CALL_QUOTE);
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,589.50", {
    timeout: 10_000,
  });

  // Authoritative snapshot B arrives (e.g. portfolio.reconciled) without a new tick.
  snapshotId = "snap-B";
  await sseEmit(page, {
    event_type: "portfolio.reconciled",
    payload: { reconcile: true },
  });

  // Snapshot B displayed — not the pre-B live mark (1589.50).
  await expect(callLeg.locator(".pnl-metric")).toContainText("$1,762.00", {
    timeout: 10_000,
  });
  await expect(strangleRow.locator(".pnl-metric")).toContainText("$1,550.00");
  await expect(muHeader.locator(".symbol-group-pnl")).toContainText("$1,550");
  await expect(callLeg.locator(".pnl-metric")).not.toContainText("$1,589.50");
});
