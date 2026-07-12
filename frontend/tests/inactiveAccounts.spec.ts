import { expect, test, type Page, type Route } from "@playwright/test";

type MockAccount = {
  account_id: string;
  label: string;
  account_type: string;
  net_liquidating_value: number;
  cash_balance: number;
  buying_power: number;
  maintenance_excess?: number | null;
  day_trading_buying_power?: number | null;
  pnl_today: number;
  positions: unknown[];
};

type MockStrategy = {
  strategy_id: string;
  account_id: string;
  underlying: string;
  strategy_type: string;
  expiration_date: string | null;
  days_to_expiration: number | null;
  quantity: number;
  strikes: string;
  unrealized_pnl: number;
  unrealized_pnl_percent: number | null;
  total_delta: number;
  total_theta: number;
  horizon: string;
  legs: unknown[];
};

const ACTIVE: MockAccount = {
  account_id: "acct-active",
  label: "Active IRA",
  account_type: "Roth IRA",
  net_liquidating_value: 25000,
  cash_balance: 5000,
  buying_power: 10000,
  maintenance_excess: 8000,
  day_trading_buying_power: 20000,
  pnl_today: 12,
  positions: [{ symbol: "SPY" }],
};

const EMPTY: MockAccount = {
  account_id: "acct-empty",
  label: "Empty Joint",
  account_type: "Joint",
  net_liquidating_value: 0,
  cash_balance: 0,
  buying_power: 0,
  maintenance_excess: null,
  day_trading_buying_power: null,
  pnl_today: 0,
  positions: [],
};

const RESIDUAL: MockAccount = {
  account_id: "acct-residual",
  label: "Residual Margin",
  account_type: "Margin",
  net_liquidating_value: 42.5,
  cash_balance: 15,
  buying_power: 10,
  maintenance_excess: 5,
  day_trading_buying_power: 0,
  pnl_today: 3,
  positions: [],
};

const THRESHOLD: MockAccount = {
  account_id: "acct-threshold",
  label: "Threshold Cash",
  account_type: "Individual",
  net_liquidating_value: 100,
  cash_balance: 100,
  buying_power: 0,
  maintenance_excess: null,
  day_trading_buying_power: null,
  pnl_today: 0,
  positions: [],
};

const DEBT: MockAccount = {
  account_id: "acct-debt",
  label: "Debt Margin",
  account_type: "Margin",
  net_liquidating_value: -250,
  cash_balance: -250,
  buying_power: 0,
  maintenance_excess: 0,
  day_trading_buying_power: 0,
  pnl_today: -5,
  positions: [],
};

const ZERO_WITH_STRATEGY: MockAccount = {
  account_id: "acct-strat-only",
  label: "Strategy Only",
  account_type: "Individual",
  net_liquidating_value: 0,
  cash_balance: 0,
  buying_power: 0,
  maintenance_excess: null,
  day_trading_buying_power: null,
  pnl_today: 0,
  positions: [],
};

const STRATEGY_FOR_ZERO: MockStrategy = {
  strategy_id: "strat-zero",
  account_id: "acct-strat-only",
  underlying: "QQQ",
  strategy_type: "Short Put",
  expiration_date: "2026-08-21",
  days_to_expiration: 21,
  quantity: 1,
  strikes: "$400",
  unrealized_pnl: 10,
  unrealized_pnl_percent: 5,
  total_delta: -5,
  total_theta: 1,
  horizon: "tactical",
  legs: [],
};

const STRATEGY_ACTIVE: MockStrategy = {
  strategy_id: "strat-active",
  account_id: "acct-active",
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
};

const SERVER_TOTALS = {
  net_liquidating_value: 24992.5,
  cash_balance: 4915,
  buying_power: 10010,
  unrealized_pnl: 50,
};

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

async function mockBaseApis(page: Page, primaryAccountId = "all") {
  await page.route("**/api/v1/session/exchange", (route) => route.fulfill({ status: 204 }));
  await page.route("**/api/v1/events", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: ": inactive-accounts heartbeat\n\n",
    }),
  );
  await page.route("**/api/v1/bootstrap", (route) =>
    fulfillJson(route, {
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
      primary_account_id: primaryAccountId,
      data_state: "ready",
      server_time: "2026-07-11T16:30:00Z",
    }),
  );
  await page.route("**/api/v1/alerts**", (route) => fulfillJson(route, []));
  await page.route("**/api/v1/recommendations**", (route) => fulfillJson(route, []));
  await page.route("**/api/v1/monitoring**", (route) =>
    fulfillJson(route, {
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
  );
  await page.route("**/api/v1/markets", (route) =>
    fulfillJson(route, {
      captured_at: "2026-07-11T16:30:00Z",
      quotes: [],
      iv_summary: {},
    }),
  );
  await page.route("**/api/v1/watchlist", (route) =>
    fulfillJson(route, { symbols: [], quotes: [] }),
  );
  await page.route("**/api/v1/streaming/status", (route) =>
    fulfillJson(route, {
      market: { state: "disabled", error: null },
      account: { state: "disabled", error: null },
    }),
  );
  await page.route("**/api/v1/catalysts**", (route) =>
    fulfillJson(route, {
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
  );
  await page.route("**/api/v1/portfolio/risk**", (route) =>
    fulfillJson(route, {
      total_delta: -20,
      total_gamma: 0.1,
      total_theta: 4,
      total_vega: -8,
      unrealized_pnl: SERVER_TOTALS.unrealized_pnl,
      net_liquidating_value: SERVER_TOTALS.net_liquidating_value,
      account_count: 6,
      strategy_count: 2,
      position_count: 1,
      concentration: [],
      stress: [],
    }),
  );
  await page.route("**/api/v1/settings/primary-account**", (route) =>
    route.fulfill({ status: 204 }),
  );
  await page.route("**/api/v1/accounts/**", (route) => fulfillJson(route, []));
}

function portfolioPayload(
  accounts: MockAccount[],
  strategies: MockStrategy[],
  selectedAccountId: string,
) {
  return {
    schema_version: 1,
    snapshot_id: `snapshot-${selectedAccountId}`,
    captured_at: "2026-07-11T16:30:00Z",
    state: "live",
    freshness: {
      as_of: "2026-07-11T16:30:00Z",
      provider: "tastytrade",
      state: "fresh",
    },
    accounts,
    strategies,
    totals: SERVER_TOTALS,
    selected_account_id: selectedAccountId,
    notice: null,
  };
}

async function mockPortfolio(
  page: Page,
  allAccounts: MockAccount[],
  allStrategies: MockStrategy[],
) {
  await page.route("**/api/v1/portfolio**", async (route) => {
    if (route.request().url().includes("/portfolio/risk")) {
      return route.fallback();
    }
    const selected =
      new URL(route.request().url()).searchParams.get("account_id") ?? "all";
    if (selected === "all") {
      return fulfillJson(route, portfolioPayload(allAccounts, allStrategies, "all"));
    }
    const accounts = allAccounts.filter((row) => row.account_id === selected);
    const strategies = allStrategies.filter((row) => row.account_id === selected);
    return fulfillJson(route, portfolioPayload(accounts, strategies, selected));
  });
}

async function openDashboard(page: Page) {
  await page.goto("/");
  await expect(page.getByLabel("Account scope")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();
}

function accountMetric(page: Page) {
  return page.locator('.metric-rail div').filter({ hasText: "Accounts" }).locator("strong");
}

function balanceTable(page: Page) {
  return page.getByRole("region", { name: "Account balances table" });
}

test.describe("inactive account presentation filter", () => {
  test("hides empty/residual accounts; keeps threshold, debt, positions, strategies", async ({
    page,
  }) => {
    const allAccounts = [ACTIVE, EMPTY, RESIDUAL, THRESHOLD, DEBT, ZERO_WITH_STRATEGY];
    const allStrategies = [STRATEGY_ACTIVE, STRATEGY_FOR_ZERO];
    await mockBaseApis(page, "all");
    await mockPortfolio(page, allAccounts, allStrategies);
    await openDashboard(page);

    const scope = page.getByLabel("Account scope");
    const optionLabels = await scope.locator("option").allTextContents();
    expect(optionLabels).toEqual([
      "All accounts",
      "Active IRA",
      "Threshold Cash",
      "Debt Margin",
      "Strategy Only",
    ]);
    expect(optionLabels.join("\n")).not.toContain("Empty Joint");
    expect(optionLabels.join("\n")).not.toContain("Residual Margin");

    await expect(accountMetric(page)).toHaveText("4");

    const table = balanceTable(page);
    await expect(table.getByRole("row", { name: /Active IRA/ })).toBeVisible();
    await expect(table.getByRole("row", { name: /Threshold Cash/ })).toBeVisible();
    await expect(table.getByRole("row", { name: /Debt Margin/ })).toBeVisible();
    await expect(table.getByRole("row", { name: /Strategy Only/ })).toBeVisible();
    await expect(table.getByText("Empty Joint")).toHaveCount(0);
    await expect(table.getByText("Residual Margin")).toHaveCount(0);

    // Server totals remain authoritative — not recomputed from the filtered list.
    await expect(page.locator(".portfolio-total strong")).toHaveText(/\$24,992\.50|\$24992\.50/);
  });

  test("all-filtered snapshot shows empty state while All accounts remains", async ({ page }) => {
    await mockBaseApis(page, "all");
    await mockPortfolio(page, [EMPTY, RESIDUAL], []);
    await openDashboard(page);

    const scope = page.getByLabel("Account scope");
    await expect(scope.locator("option")).toHaveCount(1);
    await expect(scope.locator("option")).toHaveText("All accounts");
    await expect(scope).toHaveValue("all");

    await expect(accountMetric(page)).toHaveText("0");
    await expect(balanceTable(page).getByText("No active accounts")).toBeVisible();
    await expect(page.locator(".portfolio-total strong")).toHaveText(/\$24,992\.50|\$24992\.50/);
  });

  test("explicit inactive selection preserves the option and does not mislabel scope", async ({
    page,
  }) => {
    const allAccounts = [ACTIVE, EMPTY];
    const allStrategies = [STRATEGY_ACTIVE];
    await mockBaseApis(page, "acct-empty");
    await mockPortfolio(page, allAccounts, allStrategies);
    await openDashboard(page);

    const scope = page.getByLabel("Account scope");
    // Explicit scope from server primary remains selected and present as an option.
    await expect(scope).toHaveValue("acct-empty");
    const optionLabels = await scope.locator("option").allTextContents();
    expect(optionLabels).toContain("Empty Joint");
    expect(optionLabels).toContain("Active IRA");
    expect(optionLabels).toContain("All accounts");

    // Scoped snapshot preserves the inactive account in the table for this view.
    await expect(balanceTable(page).getByRole("row", { name: /Empty Joint/ })).toBeVisible();
    await expect(accountMetric(page)).toHaveText("1");

    // Selecting All accounts again drops the inactive option from the normal filter.
    await scope.selectOption("all");
    await expect(scope).toHaveValue("all");
    await expect
      .poll(async () => scope.locator("option").allTextContents())
      .toEqual(["All accounts", "Active IRA"]);
    await expect(balanceTable(page).getByText("Empty Joint")).toHaveCount(0);
    await expect(accountMetric(page)).toHaveText("1");
  });
});
