import { defineConfig } from "@playwright/test";

const baseURL = process.env.POSITION_PILOT_BASE_URL ?? "http://127.0.0.1:8765";
const managedServer = !process.env.POSITION_PILOT_BASE_URL;

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.02,
    },
  },
  use: {
    baseURL,
    browserName: "chromium",
    reducedMotion: "reduce",
  },
  // When POSITION_PILOT_BASE_URL is unset, serve the built static frontend locally.
  ...(managedServer
    ? {
        webServer: {
          command: "pnpm exec vite preview --host 127.0.0.1 --port 8765 --strictPort",
          url: "http://127.0.0.1:8765",
          reuseExistingServer: true,
          timeout: 120_000,
        },
      }
    : {}),
});
