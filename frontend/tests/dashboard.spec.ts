import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

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

  // Combined multi-leg strategies render as one row each (not one row per leg).
  await expect(page.getByRole("button", { name: /Expand legs for QQQ Iron Condor/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Expand legs for AAPL Covered Call/i })).toBeVisible();
  await expect(page.getByText("4 legs").first()).toBeVisible();
  await expect(page.getByText("2 legs").first()).toBeVisible();
  // Single-leg rows open analysis directly and have no false leg disclosure.
  await expect(page.getByRole("button", { name: "Open SPY Short Put" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Expand legs for SPY Short Put/i })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Expand legs for SPY Long Stock/i })).toHaveCount(0);
  await expect(
    page.getByText(/Multi-leg rows are Position Pilot detected strategies/i),
  ).toBeVisible();

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

test("combined strategy leg disclosure: expand, independence, filters, analysis, a11y", async ({
  page,
}) => {
  await mockDashboardApis(page);
  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);
  await page.getByRole("button", { name: "Positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();

  const qqqToggle = page.getByRole("button", { name: /Expand legs for QQQ Iron Condor/i });
  const aaplToggle = page.getByRole("button", { name: /Expand legs for AAPL Covered Call/i });
  const qqqPanel = page.locator("#strategy-legs-panel-strat-qqq");
  const aaplPanel = page.locator("#strategy-legs-panel-strat-aapl-cc");
  await expect(qqqToggle).toHaveAttribute("aria-expanded", "false");
  await expect(aaplToggle).toHaveAttribute("aria-expanded", "false");
  await expect(qqqToggle).toHaveAttribute("aria-controls", "strategy-legs-panel-strat-qqq");
  await expect(aaplToggle).toHaveAttribute("aria-controls", "strategy-legs-panel-strat-aapl-cc");
  // Collapsed: aria-controls targets stay in the DOM but are hidden/inert.
  await expect(qqqPanel).toBeAttached();
  await expect(qqqPanel).toBeHidden();
  await expect(aaplPanel).toBeAttached();
  await expect(aaplPanel).toBeHidden();
  await expect(qqqPanel.locator("table")).toHaveCount(0);
  await expect(page.getByRole("region", { name: /Legs for QQQ Iron Condor/i })).toHaveCount(0);
  await expect(page.getByRole("region", { name: /Legs for AAPL Covered Call/i })).toHaveCount(0);

  // Expand Iron Condor: same panel id becomes visible with four option legs; other stays collapsed.
  await qqqToggle.click();
  await expect(
    page.getByRole("button", { name: /Collapse legs for QQQ Iron Condor/i }),
  ).toHaveAttribute("aria-expanded", "true");
  await expect(aaplToggle).toHaveAttribute("aria-expanded", "false");
  await expect(qqqPanel).toBeVisible();
  await expect(qqqPanel).toHaveAttribute("id", "strategy-legs-panel-strat-qqq");
  await expect(aaplPanel).toBeHidden();
  const qqqLegs = page.getByRole("region", { name: /Legs for QQQ Iron Condor/i });
  await expect(qqqLegs).toBeVisible();
  await expect(page.getByRole("region", { name: /Legs for AAPL Covered Call/i })).toHaveCount(0);
  const qqqLegRows = qqqLegs.locator("tbody tr");
  await expect(qqqLegRows).toHaveCount(4);
  await expect(qqqLegs).toContainText("Long 1");
  await expect(qqqLegs).toContainText("Short 1");
  await expect(qqqLegs).toContainText("Put");
  await expect(qqqLegs).toContainText("Call");
  await expect(qqqLegs).toContainText("$470");
  await expect(qqqLegs).toContainText("$475");
  await expect(qqqLegs).toContainText("$505");
  await expect(qqqLegs).toContainText("$510");
  await expect(qqqLegs).toContainText("$0.40");
  await expect(qqqLegs).toContainText("$1.10");
  await expect(qqqLegs).toContainText("$1.20");
  await expect(qqqLegs).toContainText("$0.50");
  await expect(qqqLegs).toContainText("QQQ  260815P00470000");
  // No account identifiers in the leg ledger.
  await expect(qqqLegs).not.toContainText("public-account-id");

  // Expand Covered Call independently: Equity + Call legs; currency marks.
  await aaplToggle.click();
  await expect(
    page.getByRole("button", { name: /Collapse legs for AAPL Covered Call/i }),
  ).toHaveAttribute("aria-expanded", "true");
  await expect(aaplPanel).toBeVisible();
  await expect(aaplPanel).toHaveAttribute("id", "strategy-legs-panel-strat-aapl-cc");
  const aaplLegs = page.getByRole("region", { name: /Legs for AAPL Covered Call/i });
  await expect(aaplLegs).toBeVisible();
  await expect(aaplLegs.locator("tbody tr")).toHaveCount(2);
  await expect(aaplLegs).toContainText("Long 100");
  await expect(aaplLegs).toContainText("Equity");
  await expect(aaplLegs).toContainText("Short 1");
  await expect(aaplLegs).toContainText("Call");
  await expect(aaplLegs).toContainText("$220");
  await expect(aaplLegs).toContainText("$210.00");
  await expect(aaplLegs).toContainText("$1.50");
  await expect(aaplLegs).toContainText("AAPL 260821C00220000");
  await expect(aaplLegs).not.toContainText("public-account-id");
  // QQQ still expanded (independent state).
  await expect(qqqLegs).toBeVisible();
  await expect(qqqPanel).toBeVisible();

  // Collapse QQQ legs; panel id remains but is hidden again; AAPL remains open.
  await page.getByRole("button", { name: /Collapse legs for QQQ Iron Condor/i }).click();
  await expect(
    page.getByRole("button", { name: /Expand legs for QQQ Iron Condor/i }),
  ).toHaveAttribute("aria-expanded", "false");
  await expect(qqqPanel).toBeAttached();
  await expect(qqqPanel).toBeHidden();
  await expect(qqqPanel.locator("table")).toHaveCount(0);
  await expect(page.getByRole("region", { name: /Legs for QQQ Iron Condor/i })).toHaveCount(0);
  await expect(aaplLegs).toBeVisible();

  // Symbol collapse hides nested ledger but preserves expanded choice.
  const aaplSymbolToggle = page.locator("button.symbol-group-toggle").filter({ hasText: "AAPL" });
  await aaplSymbolToggle.click();
  await expect(aaplSymbolToggle).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByRole("region", { name: /Legs for AAPL Covered Call/i })).toHaveCount(0);
  await aaplSymbolToggle.click();
  await expect(aaplSymbolToggle).toHaveAttribute("aria-expanded", "true");
  await expect(
    page.getByRole("button", { name: /Collapse legs for AAPL Covered Call/i }),
  ).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByRole("region", { name: /Legs for AAPL Covered Call/i })).toBeVisible();

  // Stock/Options filter: Covered Call stays under Options; nested state preserved when shown.
  const stockToggle = page.getByRole("checkbox", { name: /Show stock positions/i });
  const optionsToggle = page.getByRole("checkbox", { name: /Show options positions/i });
  await stockToggle.uncheck();
  await expect(
    page.getByRole("button", { name: /Collapse legs for AAPL Covered Call/i }),
  ).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByRole("region", { name: /Legs for AAPL Covered Call/i })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toHaveCount(0);

  await optionsToggle.uncheck();
  await stockToggle.check();
  await expect(
    page.getByRole("button", { name: /Expand legs for AAPL Covered Call|Collapse legs for AAPL Covered Call/i }),
  ).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Open SPY Long Stock" })).toBeVisible();
  // Re-enable Options: nested expanded state for AAPL restored.
  await optionsToggle.check();
  await expect(
    page.getByRole("button", { name: /Collapse legs for AAPL Covered Call/i }),
  ).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByRole("region", { name: /Legs for AAPL Covered Call/i })).toBeVisible();

  // Analysis action opens the strategy drawer without depending on disclosure.
  await page.getByRole("button", { name: "Open analysis for AAPL Covered Call" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();

  // Single-leg still opens analysis directly.
  await page.getByRole("button", { name: "Open SPY Short Put" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");

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
  const wideAxe = await new AxeBuilder({ page }).analyze();
  expect(wideAxe.violations).toEqual([]);
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
    // Body text: ≥16px on phone; desktop/tablet may stay denser (≥13px).
    if (width <= 390) {
      expect(metrics.bodyFontSize).toBeGreaterThanOrEqual(16);
    } else {
      expect(metrics.bodyFontSize).toBeGreaterThanOrEqual(13);
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
