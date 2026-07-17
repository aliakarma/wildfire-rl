import { defineConfig } from "@playwright/test";

/**
 * Accessibility matrix (Dashboard_Guide.md §13/§16): axe-core over every route in both
 * locales and both themes, against the production build served by `vite preview`.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: "http://localhost:4173" },
  webServer: {
    command: "npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
  },
});
