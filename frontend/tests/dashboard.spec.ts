import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("secure dashboard shell renders without console or accessibility errors", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  const launchToken = process.env.POSITION_PILOT_LAUNCH_TOKEN ?? "browser-smoke-launch";
  await page.goto(`/?launch_token=${encodeURIComponent(launchToken)}`);

  await expect(page.getByRole("heading", { name: "Decision field" })).toBeVisible();
  await expect(page).not.toHaveURL(/launch_token/);
  await expect(page.getByRole("complementary", { name: "Primary navigation" })).toBeVisible();
  await expect(page.getByText("Provider ledger")).toBeVisible();

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
