import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: process.env.POSITION_PILOT_BASE_URL ?? "http://127.0.0.1:8765",
    browserName: "chromium",
  },
});
