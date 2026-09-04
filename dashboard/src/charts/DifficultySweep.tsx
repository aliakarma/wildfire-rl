import { useTranslation } from "react-i18next";

import type { TableModel } from "../components/ChartCard";
import type { Region, Robustness } from "../lib/data";
import { fmtWEL } from "../lib/format";
import { methodColor } from "../theme/echartsTheme";
import { token } from "../theme/useTheme";
import { EChart, type EChartsOption } from "./EChart";

/** Canonical policy order (Table 1); the chart renders whichever of these the artifact carries. */
const POLICY_ORDER = [
  "noop",
  "value_first",
  "greedy_risk",
  "local_reactive",
  "mappo",
  "commnet",
  "hiercomm_heur",
];
const REGIMES = ["easy", "medium", "hard"];

/** Policies present in every regime of this region's block, in canonical order. */
function presentPolicies(block: Robustness["regions"][Region]): string[] {
  return POLICY_ORDER.filter((p) => REGIMES.every((r) => p in (block[r] ?? {})));
}

interface Props {
  robustness: Robustness;
  region: Region;
}

/** Difficulty sweep (§6.5): grouped WEL bars across easy/medium/hard; ghost-bar No-Op anchor. */
export function DifficultySweep({ robustness, region }: Props) {
  const { t } = useTranslation();
  const block = robustness.regions[region];
  const POLICIES = presentPolicies(block);
  const textSecondary = token("--text-secondary");
  const textPrimary = token("--text-primary");
  const border = token("--border");

  const option: EChartsOption = {
    grid: { left: 8, right: 16, top: 64, bottom: 24, containLabel: true },
    legend: {
      data: POLICIES.map((p) => t(`policy.${p}`)),
      textStyle: { color: textSecondary, fontSize: 12 },
      top: 0,
      left: 0,
    },
    xAxis: {
      type: "category",
      data: REGIMES.map((r) => t(`difficulty.${r}`)),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: textSecondary, fontSize: 12 },
    },
    yAxis: {
      type: "value",
      name: "WEL",
      nameTextStyle: { color: textSecondary, fontSize: 12 },
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: border } },
      axisLabel: { color: textSecondary, fontSize: 12 },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: token("--surface-1"),
      borderColor: border,
      textStyle: { color: textPrimary, fontSize: 12 },
      valueFormatter: (v: unknown) => (typeof v === "number" ? fmtWEL(v) : String(v)),
    },
    series: POLICIES.map((p) => ({
      type: "bar",
      name: t(`policy.${p}`),
      data: REGIMES.map((r) => block[r][p].WEL_mean),
      barMaxWidth: 24,
      itemStyle:
        p === "noop"
          ? {
              color: "transparent",
              borderColor: textSecondary,
              borderWidth: 1.5,
              borderType: "dashed" as const,
              borderRadius: [4, 4, 0, 0],
            }
          : { color: methodColor(p), borderRadius: [4, 4, 0, 0] },
      label: {
        show: true,
        position: "top",
        color: textPrimary,
        fontSize: 11,
        fontFamily: "IBM Plex Mono, monospace",
        formatter: ({ value }: { value: number }) => fmtWEL(value),
      },
    })),
  };

  return <EChart option={option} height={340} ariaLabel={t("robustness.sweep")} />;
}

export function robustnessTable(
  robustness: Robustness,
  region: Region,
  t: (k: string) => string,
): TableModel {
  const block = robustness.regions[region];
  const POLICIES = presentPolicies(block);
  const rows = REGIMES.flatMap((r) =>
    POLICIES.map((p) => ({
      display: [
        t(`difficulty.${r}`),
        t(`policy.${p}`),
        fmtWEL(block[r][p].WEL_mean),
        block[r][p].ISR_mean.toFixed(3),
      ],
      csv: [r, p, block[r][p].WEL_mean, block[r][p].ISR_mean],
    })),
  );
  return {
    columns: [
      { header: t("filters.difficulty") },
      { header: t("results.baseline") },
      { header: "WEL", numeric: true },
      { header: "ISR", numeric: true },
    ],
    rows: rows.map((r) => r.display),
    csvHeaders: ["difficulty", "policy", "WEL_mean", "ISR_mean"],
    csvRows: rows.map((r) => r.csv),
  };
}
