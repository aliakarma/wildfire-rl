import { useTranslation } from "react-i18next";

import type { TableModel } from "../components/ChartCard";
import type { Ablations, Region } from "../lib/data";
import { fmtP, fmtWEL, sigStars } from "../lib/format";
import { token } from "../theme/useTheme";
import { EChart, type EChartsOption } from "./EChart";

interface Props {
  ablations: Ablations;
  region: Region;
}

const VARIANT_ORDER = ["wo_learned_tactic", "wo_hierarchy", "wo_rl_finetune", "wo_comms", "wo_shaping"];

/**
 * Component-contribution dumbbell (§6.4): Full HierComm WEL vs each ablated variant,
 * sorted by effect. Ablation variants wear HierComm orange at 55% opacity — they are
 * "HierComm minus a part", never new entities (§7.3). Non-significant variants carry
 * an explicit n.s. label.
 */
export function AblationDumbbell({ ablations, region }: Props) {
  const { t } = useTranslation();
  const block = ablations.regions[region];
  const full = block.full;
  const accent = token("--m-hiercomm");
  const textPrimary = token("--text-primary");
  const textSecondary = token("--text-secondary");
  const border = token("--border");

  const variants = VARIANT_ORDER.filter((v) => v in block).sort(
    (a, b) =>
      Math.abs(block[b].WEL_mean - full.WEL_mean) - Math.abs(block[a].WEL_mean - full.WEL_mean),
  );

  const labels = variants.map((v) => block[v].label);

  const option: EChartsOption = {
    grid: { left: 8, right: 110, top: 8, bottom: 24, containLabel: true },
    xAxis: {
      type: "value",
      name: "WEL",
      nameLocation: "middle",
      nameGap: 26,
      nameTextStyle: { color: textSecondary, fontSize: 12 },
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
      axisLabel: { color: textSecondary, fontSize: 12, width: 180, overflow: "break" },
    },
    tooltip: {
      trigger: "item",
      backgroundColor: token("--surface-1"),
      borderColor: border,
      textStyle: { color: textPrimary, fontSize: 12 },
      formatter: ({ dataIndex }: { dataIndex: number }) => {
        const v = block[variants[dataIndex]];
        const lines = [
          `<b>${v.label}</b>`,
          `WEL: ${fmtWEL(v.WEL_mean)} (${t("policy.hiercomm_heur")}: ${fmtWEL(full.WEL_mean)})`,
        ];
        if (v.vs_full_WEL) {
          // dir="ltr" keeps "p = 0.295 · d = 2.27" from bidi-scrambling in the Arabic UI.
          lines.push(
            `<span dir="ltr">${fmtP(v.vs_full_WEL.p)} · d = ${v.vs_full_WEL.d.toFixed(2)}</span>`,
          );
        } else {
          lines.push(t("ablation.deterministicNote"));
        }
        return lines.join("<br/>");
      },
    },
    series: [
      {
        type: "custom",
        renderItem: (
          params: { dataIndex: number },
          api: {
            value: (i: number) => number;
            coord: (v: [number, number]) => [number, number];
          },
        ) => {
          const i = params.dataIndex;
          const v = block[variants[i]];
          const from = api.coord([api.value(1), i]);
          const to = api.coord([api.value(2), i]);
          const ns = v.vs_full_WEL ? sigStars(v.vs_full_WEL.p) === "n.s." : false;
          const deltaText = `Δ ${fmtWEL(v.WEL_mean - full.WEL_mean)}${
            v.vs_full_WEL ? ` ${sigStars(v.vs_full_WEL.p)}` : " †"
          }`;
          return {
            type: "group",
            children: [
              {
                type: "line",
                shape: { x1: from[0], y1: from[1], x2: to[0], y2: to[1] },
                style: { stroke: accent, lineWidth: 2, opacity: 0.4 },
              },
              {
                type: "circle",
                shape: { cx: from[0], cy: from[1], r: 5 },
                style: { fill: accent },
              },
              {
                type: "circle",
                shape: { cx: to[0], cy: to[1], r: 6 },
                style: { fill: accent, opacity: 0.55 },
              },
              {
                type: "text",
                style: {
                  x: Math.max(from[0], to[0]) + 12,
                  y: to[1],
                  text: deltaText,
                  fill: ns ? textSecondary : textPrimary,
                  font: '12px "IBM Plex Mono", monospace',
                  verticalAlign: "middle",
                },
              },
            ],
          };
        },
        data: variants.map((v, i) => [i, full.WEL_mean, block[v].WEL_mean]),
        encode: { x: [1, 2], y: 0 },
      },
    ],
  };

  return <EChart option={option} height={280} ariaLabel={t("ablation.contribution")} />;
}

export function ablationTable(
  ablations: Ablations,
  region: Region,
  t: (k: string) => string,
): TableModel {
  const block = ablations.regions[region];
  const keys = ["full", ...VARIANT_ORDER.filter((v) => v in block)];
  const rows = keys.map((k) => {
    const v = block[k];
    return {
      display: [
        v.label,
        fmtWEL(v.WEL_mean),
        `[${fmtWEL(v.WEL_ci[0])}, ${fmtWEL(v.WEL_ci[1])}]`,
        v.ISR_mean.toFixed(3),
        v.vs_full_WEL ? (
          <bdi dir="ltr">{fmtP(v.vs_full_WEL.p)}</bdi>
        ) : k === "full" ? (
          "—"
        ) : (
          t("ablation.deterministicNote")
        ),
        v.vs_full_WEL ? v.vs_full_WEL.d.toFixed(2) : "—",
        String(v.n_seeds),
      ],
      csv: [
        k,
        v.WEL_mean,
        v.WEL_ci[0],
        v.WEL_ci[1],
        v.ISR_mean,
        v.vs_full_WEL?.p ?? "",
        v.vs_full_WEL?.d ?? "",
        v.n_seeds,
      ],
    };
  });
  return {
    columns: [
      { header: t("ablation.variantCol") },
      { header: "WEL", numeric: true },
      { header: t("stats.ci95"), numeric: true },
      { header: "ISR", numeric: true },
      { header: t("stats.pvalue"), numeric: true },
      { header: t("stats.effect"), numeric: true },
      { header: "n", numeric: true },
    ],
    rows: rows.map((r) => r.display),
    csvHeaders: ["variant", "WEL_mean", "WEL_ci_lo", "WEL_ci_hi", "ISR_mean", "p_vs_full", "d_vs_full", "n_seeds"],
    csvRows: rows.map((r) => r.csv),
  };
}
