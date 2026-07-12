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
            legs: [],
          },
        ],
        totals: {
          net_liquidating_value: 25000,
          cash_balance: 5000,
          buying_power: 10000,
          unrealized_pnl: 40,
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
        roll_id: "roll-public",
        timestamp: "2026-07-09T16:30:00Z",
        old_strike: 500,
        new_strike: 495,
        old_dte: 14,
        new_dte: 35,
        roll_pnl: 50,
        premium_effect: 0.3,
      },
    ],
  };
  await page.route("**/api/v1/accounts/*/rolls", (route) =>
    route.fulfill({ contentType: "application/json", body: JSON.stringify([rollChain]) }),
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
  await expect(page.getByText("Short Put")).toBeVisible();
  await expect(page.getByText("No confirmed catalyst found").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Order activity" })).toBeVisible();

  await page.getByRole("button", { name: "SPY" }).click();
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
  await expect(page.getByText("Short Put")).toBeVisible();
  const tableMetrics = await page.evaluate(() => {
    const row = document.querySelector(".data-table.dense tbody tr") as HTMLElement | null;
    const wrap = document.querySelector(".table-wrap") as HTMLElement | null;
    return {
      rowHeight: row?.getBoundingClientRect().height ?? 0,
      pageOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
      tableScrollable: wrap ? wrap.scrollWidth >= wrap.clientWidth : false,
    };
  });
  expect(tableMetrics.rowHeight).toBeGreaterThan(20);
  expect(tableMetrics.rowHeight).toBeLessThan(72);
  expect(tableMetrics.pageOverflow).toBeFalsy();
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
  await page.getByRole("button", { name: "SPY" }).click();
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
