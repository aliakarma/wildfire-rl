import { useTranslation } from "react-i18next";

import type { TableModel } from "../components/ChartCard";
import type { MainResults, Region } from "../lib/data";
import { fmt, fmtISR, fmtWEL } from "../lib/format";
import { methodColor } from "../theme/echartsTheme";
import { token } from "../theme/useTheme";
import { EChart, type EChartsOption } from "./EChart";

interface Props {
  results: MainResults;
  region: Region;
  metric: "WEL" | "ISR";
}

/**
 * WEL/ISR by policy (§6.3): horizontal bars in the fixed policy order, bootstrap 95% CI
 * whiskers + per-seed dots on learned methods, ghost-bar No-Op, direct value labels
 * (relief rule §7.3 — mandatory), "deterministic" tagging instead of fake CIs.
 */
export function ResultsBars({ results, region, metric }: Props) {
  const { t } = useTranslation();
  const order = results.policy_order;
  const policies = results.regions[region].policies;
  const fmtVal = metric === "WEL" ? fmtWEL : fmtISR;

  const labels = order.map((p) => t(`policy.${p}`));
  const textPrimary = token("--text-primary");
  const textSecondary = token("--text-secondary");
  const border = token("--border");
  const surface1 = token("--surface-1");

  const bars = order.map((p) => {
    const m = policies[p][metric];
    const noop = p === "noop";
    return {
      value: m.mean,
      itemStyle: noop
        ? {
            color: "transparent",
            borderColor: textSecondary,
            borderWidth: 1.5,
            borderType: "dashed" as const,
            borderRadius: [0, 4, 4, 0],
          }
        : { color: methodColor(p), borderRadius: [0, 4, 4, 0] },
    };
  });

  const ciData = order
    .map((p, i) => ({ p, i, m: policies[p][metric] }))
    .filter(({ p }) => policies[p].learned)
    .map(({ i, m }) => [i, m.ci[0], m.ci[1]]);

  const seedDots: [number, number][] = [];
  order.forEach((p, i) => {
    if (policies[p].learned) policies[p][metric].values.forEach((v) => seedDots.push([v, i]));
  });

  const option: EChartsOption = {
    grid: { left: 8, right: 64, top: 8, bottom: 24, containLabel: true },
    xAxis: {
      type: "value",
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: border } },
      axisLabel: { color: textSecondary, fontSize: 12 },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: textSecondary, fontSize: 12, width: 170, overflow: "break" },
    },
    tooltip: {
      trigger: "item",
      backgroundColor: surface1,
      borderColor: border,
      textStyle: { color: textPrimary, fontSize: 12 },
      formatter: (params: { dataIndex?: number; seriesType?: string; value?: unknown }) => {
        const i =
          params.seriesType === "scatter"
            ? (params.value as [number, number])[1]
            : (params.dataIndex ?? 0);
        const p = order[i];
        const m = policies[p][metric];
        const lines = [`<b>${t(`policy.${p}`)}</b>`, `${metric}: ${fmt(m.mean)}`];
        if (policies[p].learned) {
          lines.push(`${t("stats.ci95")}: [${fmtVal(m.ci[0])}, ${fmtVal(m.ci[1])}]`);
          lines.push(t("stats.nSeeds", { n: m.n_train_seeds }));
        } else {
          lines.push(t("stats.deterministic"));
        }
        return lines.join("<br/>");
      },
    },
    series: [
      {
        type: "bar",
        data: bars,
        barMaxWidth: 22,
        barCategoryGap: "35%",
        label: {
          // Learned methods carry their direct label past the CI whisker (custom
          // series below) so text never collides with whiskers or seed dots.
          show: true,
          position: "right",
          color: textPrimary,
          fontSize: 12,
          fontFamily: "IBM Plex Mono, monospace",
          formatter: ({ dataIndex }: { dataIndex: number }) => {
            const p = order[dataIndex];
            return policies[p].learned ? "" : `${fmtVal(policies[p][metric].mean)} ⚙`;
          },
        },
        z: 1,
      },
      {
        // 95% CI whiskers (learned methods only) — 1.5px, 8px caps (§8.2).
        type: "custom",
        renderItem: (
          _params: unknown,
          api: {
            value: (i: number) => number;
            coord: (v: [number, number]) => [number, number];
          },
        ) => {
          const cat = api.value(0);
          const lo = api.coord([api.value(1), cat]);
          const hi = api.coord([api.value(2), cat]);
          const cap = 4;
          const stroke = { stroke: textSecondary, lineWidth: 1.5 };
          const mean = policies[order[cat]][metric].mean;
          return {
            type: "group",
            children: [
              {
                type: "line",
                shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] },
                style: stroke,
              },
              {
                type: "line",
                shape: { x1: lo[0], y1: lo[1] - cap, x2: lo[0], y2: lo[1] + cap },
                style: stroke,
              },
              {
                type: "line",
                shape: { x1: hi[0], y1: hi[1] - cap, x2: hi[0], y2: hi[1] + cap },
                style: stroke,
              },
              {
                type: "text",
                style: {
                  x: hi[0] + 10,
                  y: hi[1],
                  text: fmtVal(mean),
                  fill: textPrimary,
                  font: '12px "IBM Plex Mono", monospace',
                  verticalAlign: "middle",
                },
              },
            ],
          };
        },
        encode: { x: [1, 2], y: 0 },
        data: ciData,
        silent: true,
        z: 3,
      },
      {
        // Per-seed dots: 6px with a 2px surface ring (§8.2).
        type: "scatter",
        data: seedDots.map(([v, i]) => ({
          value: [v, i],
          itemStyle: {
            color: methodColor(order[i]),
            borderColor: surface1,
            borderWidth: 2,
          },
        })),
        symbolSize: 8,
        z: 4,
      },
    ],
  };

  return (
    <EChart
      option={option}
      height={300}
      ariaLabel={t(metric === "WEL" ? "results.welByPolicy" : "results.isrByPolicy")}
    />
  );
}

/** Table/CSV twin of the bars (the canonical accessible alternative, §13). */
export function resultsBarsTable(
  results: MainResults,
  region: Region,
  metric: "WEL" | "ISR",
  t: (k: string, o?: Record<string, unknown>) => string,
): TableModel {
  const order = results.policy_order;
  const policies = results.regions[region].policies;
  const fmtVal = metric === "WEL" ? fmtWEL : fmtISR;
  const rows = order.map((p) => {
    const m = policies[p][metric];
    const learned = policies[p].learned;
    return {
      display: [
        t(`policy.${p}`),
        fmtVal(m.mean),
        learned ? fmtVal(m.std) : t("stats.deterministic"),
        learned ? `[${fmtVal(m.ci[0])}, ${fmtVal(m.ci[1])}]` : "—",
        learned ? m.values.map(fmtVal).join(", ") : "—",
      ],
      csv: [
        p,
        m.mean,
        learned ? m.std : "",
        learned ? m.ci[0] : "",
        learned ? m.ci[1] : "",
        learned ? m.values.join(";") : "",
      ],
    };
  });
  return {
    columns: [
      { header: t("results.baseline") },
      { header: metric, numeric: true },
      { header: "σ", numeric: true },
      { header: t("stats.ci95"), numeric: true },
      { header: t("stats.perSeed"), numeric: true },
    ],
    rows: rows.map((r) => r.display),
    csvHeaders: ["policy", `${metric}_mean`, "std", "ci_lo", "ci_hi", "per_seed_values"],
    csvRows: rows.map((r) => r.csv),
  };
}
