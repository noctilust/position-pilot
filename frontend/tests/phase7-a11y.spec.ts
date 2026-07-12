import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

async function mockCoreApis(page: Page) {
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
        application: {
          name: "Position Pilot",
          version: "test",
          phase: "hardening-retirement",
        },
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

  const empty = (route: { fulfill: (r: object) => unknown }) =>
    route.fulfill({ contentType: "application/json", body: "[]" });

  await page.route("**/api/v1/alerts**", empty);
  await page.route("**/api/v1/recommendations**", empty);
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

  const strategies = Array.from({ length: 60 }, (_, index) => ({
    strategy_id: `strat-${index}`,
    account_id: "public-account-id",
    underlying: index === 0 ? "SPY" : `U${index}`,
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
  }));

  await page.route("**/api/v1/portfolio**", (route) => {
    if (route.request().url().includes("/risk")) {
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          total_delta: -20,
          total_gamma: 0.1,
          total_theta: 4,
          total_vega: -8,
          unrealized_pnl: 40,
          net_liquidating_value: 25000,
          account_count: 1,
          strategy_count: strategies.length,
          position_count: strategies.length,
          concentration: [],
          stress: [],
        }),
      });
    }
    return route.fulfill({
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
        strategies,
        totals: {
          net_liquidating_value: 25000,
          cash_balance: 5000,
          buying_power: 10000,
          unrealized_pnl: 40,
        },
        selected_account_id: "all",
        notice: null,
      }),
    });
  });

  await page.route("**/api/v1/markets**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        captured_at: "2026-07-11T16:30:00Z",
        quotes: [],
        iv_summary: {},
      }),
    }),
  );
  await page.route("**/api/v1/watchlist", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ symbols: [], quotes: [] }),
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
        results: [],
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

  // Operations endpoints used by Settings panel
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
        settings_redacted: {},
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
          excluded: ["credential values", "raw environment variables"],
          redacted_keys: [],
          policy: "redacted",
        },
        disclaimer: "Decision support only.",
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
        updated_at: null,
        settings: {
          portfolio_snapshots_days: 365,
          catalyst_events_days: 365,
          article_metadata_days: 90,
          recommendation_history_days: 0,
          transaction_history: "indefinite",
        },
        candidates: { portfolio_snapshots: 0 },
        audit_critical_preserved: ["transaction and roll-chain history"],
        would_delete: {},
        disclaimer: "Decision support only.",
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
        : JSON.stringify([]),
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
        reversible_instructions: ["Create a backup first."],
        auto_install: false,
        note: "Updates are never installed automatically.",
        disclaimer: "Decision support only.",
      }),
    }),
  );
  await page.route("**/api/v1/accounts/**", (route) =>
    route.fulfill({ contentType: "application/json", body: "[]" }),
  );
}

test.describe("Phase 7 accessibility, keyboard, responsive, visual", () => {
  test("keyboard navigation, settings operations, axe, and visual contract", async ({ page }) => {
    await mockCoreApis(page);
    const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
    await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);

    await expect(page.getByRole("complementary", { name: "Primary navigation" })).toBeVisible();

    // Keyboard: move through nav with Tab and activate Positions
    await page.keyboard.press("Tab");
    await page.getByRole("button", { name: "Positions" }).focus();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { name: "Positions", exact: true })).toBeVisible();
    // Flat strategy pagination removed: all strategies appear in symbol groups (no silent truncation).
    await expect(page.getByRole("button", { name: "Next page" })).toHaveCount(0);
    await expect(page.getByRole("checkbox", { name: /Show stock positions/i })).toBeVisible();
    await expect(page.getByRole("checkbox", { name: /Show options positions/i })).toBeVisible();
    await expect(page.getByText(/60 strategies across/i)).toBeVisible();

    // Settings operations surface
    await page.getByRole("button", { name: "Settings" }).click();
    await expect(page.getByRole("heading", { name: "Portfolio and history" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Redacted bundle and .env checks" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Create, download, restore" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Readiness (never auto-installed)" })).toBeVisible();
    await expect(page.getByText(/decision support/i).first()).toBeVisible();

    const accessibility = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
      .analyze();
    expect(accessibility.violations).toEqual([]);

    // Responsive: narrow mobile layout still exposes settings controls
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.getByRole("button", { name: "Download portfolio CSV" })).toBeVisible();
    const narrowAxe = await new AxeBuilder({ page }).analyze();
    expect(narrowAxe.violations).toEqual([]);

    // Large desktop
    await page.setViewportSize({ width: 1680, height: 1050 });
    await expect(page.getByRole("heading", { name: "Trading-day consent" })).toBeVisible();

    // Deterministic visual contract (stable landmark + heading structure)
    await page.setViewportSize({ width: 1280, height: 800 });
    const landmarks = await page.evaluate(() => {
      return {
        main: document.querySelectorAll("main, [role='main']").length,
        complementary: document.querySelectorAll("aside, [role='complementary']").length,
        navigationButtons: Array.from(
          document.querySelectorAll("nav button, [role='navigation'] button, .navigation-rail button"),
        ).map((node) => (node.textContent || "").trim()),
      };
    });
    expect(landmarks.main).toBeGreaterThanOrEqual(1);
    expect(landmarks.complementary).toBeGreaterThanOrEqual(1);
    expect(landmarks.navigationButtons.join(" ")).toMatch(/Overview|Positions|Settings/);
  });
});
