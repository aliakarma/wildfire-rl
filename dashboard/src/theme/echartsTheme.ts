import { token } from "./useTheme";

/** Fixed method → color-token mapping (§7.3). Color follows the method everywhere. */
export const METHOD_TOKEN: Record<string, string> = {
  value_first: "--m-valuefirst",
  greedy_risk: "--m-greedy",
  local_reactive: "--m-reactive",
  mappo: "--m-mappo",
  commnet: "--m-commnet",
  hiercomm_heur: "--m-hiercomm",
};

/** Resolve a method's series color for the current theme; No-Op wears ink, never a hue. */
export function methodColor(policy: string): string {
  const t = METHOD_TOKEN[policy];
  return t ? token(t) : token("--text-secondary");
}

/**
 * Base ECharts option fragments built from the current tokens. Rebuilt on every
 * theme toggle (charts re-render from tokens; no reload — §9.2).
 */
export function baseChartTheme() {
  const textSecondary = token("--text-secondary");
  const border = token("--border");
  return {
    textStyle: { fontFamily: getComputedStyle(document.body).fontFamily, color: textSecondary },
    axisLabel: { color: textSecondary, fontSize: 12 },
    axisTitle: { color: textSecondary, fontSize: 12 },
    // Horizontal hairlines only; axis lines suppressed (§8.2).
    splitLine: { lineStyle: { color: border, width: 1 } },
    tooltip: {
      backgroundColor: token("--surface-1"),
      borderColor: border,
      textStyle: { color: token("--text-primary"), fontSize: 12 },
    },
  };
}
