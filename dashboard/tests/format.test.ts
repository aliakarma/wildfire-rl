/**
 * Render-time formatting rules (Dashboard_Guide.md §4.2, §10.2, §12): WEL rounds to 2
 * decimals, ISR to 3, p-values go scientific below 0.001, significance stars follow the
 * badge spec, and the Arabic locale keeps Western digits so values match the paper.
 */
import { describe, expect, it } from "vitest";

import { setLocale } from "../src/i18n";
import { fmt, fmtISR, fmtP, fmtWEL, sigStars } from "../src/lib/format";

describe("rounding rules", () => {
  it("WEL renders with exactly 2 decimals", () => {
    expect(fmtWEL(7.023999999999999)).toBe("7.02");
    expect(fmtWEL(27)).toBe("27.00");
  });

  it("ISR renders with exactly 3 decimals", () => {
    expect(fmtISR(0.8087619047619047)).toBe("0.809");
    expect(fmtISR(0)).toBe("0.000");
  });
});

describe("p-value formatting", () => {
  it("uses scientific notation below 0.001", () => {
    expect(fmtP(0.000045948562607325285)).toBe("p = 4.6e-5");
    expect(fmtP(0.0009)).toBe("p = 9.0e-4");
  });

  it("uses plain decimals at or above 0.001", () => {
    expect(fmtP(0.3192930854233641)).toBe("p = 0.319");
    expect(fmtP(0.048)).toBe("p = 0.048");
  });
});

describe("significance stars (§12)", () => {
  it("maps thresholds to badges", () => {
    expect(sigStars(0.0005)).toBe("***");
    expect(sigStars(0.005)).toBe("**");
    expect(sigStars(0.03)).toBe("*");
    expect(sigStars(0.05)).toBe("n.s.");
    expect(sigStars(0.32)).toBe("n.s.");
  });

  it("boundaries are exclusive", () => {
    expect(sigStars(0.001)).toBe("**");
    expect(sigStars(0.01)).toBe("*");
  });
});

describe("Arabic locale keeps Western digits (§10.2)", () => {
  it("never emits Arabic-Indic digits and never changes the numeric value", () => {
    setLocale("ar");
    try {
      const rendered = fmtWEL(7.024);
      expect(rendered).not.toMatch(/[٠-٩]/);
      expect(rendered).toContain("7");
      expect(rendered).toContain("02");
      expect(fmt(1234)).not.toMatch(/[٠-٩]/);
    } finally {
      setLocale("en");
    }
  });
});
