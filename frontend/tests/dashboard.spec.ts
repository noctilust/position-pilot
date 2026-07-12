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
      "Position Pilot — Overview",
    ]),
  );

  // Real deterministic screenshot baseline (animations reduced; maxDiffPixelRatio in config).
  await expect(page.locator(".workspace-main")).toHaveScreenshot("overview-workspace.png", {
    animations: "disabled",
  });
});
