/**
 * axe-core accessibility matrix (Dashboard_Guide.md §13, §16.2): every route × locale
 * (en/ar) × theme (light/dark) must have zero WCAG 2.1 A/AA violations.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const ROUTES = [
  "/",
  "/replays",
  "/results",
  "/ablation",
  "/robustness",
  "/generalization",
  "/benchmark",
  "/method",
  "/reproducibility",
  "/about",
];
const LOCALES = ["en", "ar"] as const;
const THEMES = ["light", "dark"] as const;

for (const route of ROUTES) {
  for (const lang of LOCALES) {
    for (const theme of THEMES) {
      test(`axe ${route} [${lang}/${theme}]`, async ({ page }) => {
        await page.goto(`/#${route}?lang=${lang}&theme=${theme}&region=saudi`);
        // Data pages render after their JSON loads; wait for main content to settle.
        await page.waitForLoadState("networkidle");
        await page.waitForSelector("main h1", { timeout: 30_000 });

        const results = await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
          .analyze();

        expect(
          results.violations.map((v) => ({
            id: v.id,
            impact: v.impact,
            nodes: v.nodes.map((n) => n.target).slice(0, 5),
          })),
        ).toEqual([]);
      });
    }
  }
}
