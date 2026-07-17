/**
 * Bilingual parity guard (Dashboard_Guide.md §10): the Arabic locale is first-class, so
 * both resource files must expose exactly the same keys, and every interpolated
 * placeholder ({{x}}) and markup tag (<bdi>) present in one language must exist in the
 * other — a missing placeholder would silently drop a *number* from the translated copy.
 */
import { describe, expect, it } from "vitest";

import ar from "../src/i18n/locales/ar.json";
import en from "../src/i18n/locales/en.json";

type Tree = { [k: string]: string | Tree };

function leaves(node: Tree, prefix = ""): Map<string, string> {
  const out = new Map<string, string>();
  for (const [key, value] of Object.entries(node)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "string") out.set(path, value);
    else for (const [p, v] of leaves(value, path)) out.set(p, v);
  }
  return out;
}

const enLeaves = leaves(en as Tree);
const arLeaves = leaves(ar as Tree);

describe("locale key parity", () => {
  it("en and ar expose identical key sets", () => {
    expect([...arLeaves.keys()].sort()).toEqual([...enLeaves.keys()].sort());
  });

  it("no locale string is empty", () => {
    for (const [path, value] of [...enLeaves, ...arLeaves]) {
      expect(value.trim(), path).not.toBe("");
    }
  });
});

describe("interpolation parity", () => {
  it("every {{placeholder}} matches across languages", () => {
    const holes = (s: string) => (s.match(/{{\s*\w+\s*}}/g) ?? []).map((h) => h.replace(/\s/g, "")).sort();
    for (const [path, enValue] of enLeaves) {
      const arValue = arLeaves.get(path)!;
      expect(holes(arValue), path).toEqual(holes(enValue));
    }
  });

  it("every <tag> used by <Trans> matches across languages", () => {
    const tags = (s: string) => (s.match(/<\/?\w+>/g) ?? []).sort();
    for (const [path, enValue] of enLeaves) {
      expect(tags(arLeaves.get(path)!), path).toEqual(tags(enValue));
    }
  });
});
