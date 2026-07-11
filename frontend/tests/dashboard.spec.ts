import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("secure dashboard shell renders without console or accessibility errors", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
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
        strategies: [],
        totals: {
          net_liquidating_value: 25000,
          cash_balance: 5000,
          buying_power: 10000,
          unrealized_pnl: 0,
        },
        selected_account_id: "all",
        notice: null,
      }),
    }),
  );

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);

  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();
  await expect(page).not.toHaveURL(/launch_token/);
  await expect(page.getByRole("complementary", { name: "Primary navigation" })).toBeVisible();
  await expect(page.getByText("Provider ledger")).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Account scope" })).toBeVisible();
  await expect(page.getByText("$25,000.00")).toBeVisible();

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
