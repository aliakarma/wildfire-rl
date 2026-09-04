import { useTranslation } from "react-i18next";

import type { TableModel } from "../components/ChartCard";
import type { Generalization, Region } from "../lib/data";
import { fmtWEL } from "../lib/format";
import { methodColor } from "../theme/echartsTheme";
import { token } from "../theme/useTheme";
import { EChart, type EChartsOption } from "./EChart";

const CONDITIONS = ["standard", "heldout_ignition", "wind_plus90", "assets_rot90", "cross_region"];
const POLICIES = ["value_first", "local_reactive", "commnet", "hiercomm_heur"];

interface Props {
  generalization: Generalization;
  region: Region;
}

/**
 * Under `cross_region` the artifact keys transferred policies as
 * `<policy>_from_<other-region>`; native keys are used everywhere else.
 */
function cell(
  block: Generalization["regions"][Region],
  condition: string,
  policy: string,
  region: Region,
) {
  const other = region === "saudi" ? "california" : "saudi";
  return block[condition][policy] ?? block[condition][`${policy}_from_${other}`];
}

/**
 * ΔWEL vs No-Op per condition (§6.6) with the zero line emphasized — the falsifiability
 * story: a policy adding no value sits at 0.
 */
export function DeltaWelBars({ generalization, region }: Props) {
  const { t } = useTranslation();
  const block = generalization.regions[region];
  const textSecondary = token("--text-secondary");
  const textPrimary = token("--text-primary");
  const border = token("--border");

  const conditions = CONDITIONS.filter((c) => c in block);

  const option: EChartsOption = {
    grid: { left: 8, right: 16, top: 64, bottom: 40, containLabel: true },
    legend: {
      data: POLICIES.map((p) => t(`policy.${p}`)),
      textStyle: { color: textSecondary, fontSize: 12 },
      top: 0,
      left: 0,
    },
    xAxis: {
      type: "category",
      data: conditions.map((c) => t(`condition.${c}`)),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: textSecondary, fontSize: 11, interval: 0, width: 110, overflow: "break" },
    },
    yAxis: {
      type: "value",
      name: "ΔWEL",
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
    series: POLICIES.map((p, pi) => ({
      type: "bar",
      name: t(`policy.${p}`),
      data: conditions.map((c) => cell(block, c, p, region)?.dWEL ?? null),
      barMaxWidth: 18,
      itemStyle: { color: methodColor(p), borderRadius: [4, 4, 0, 0] },
      // Zero line emphasized once (on the first series).
      markLine:
        pi === 0
          ? {
              silent: true,
              symbol: "none",
              lineStyle: { color: textPrimary, width: 1.5, type: "solid" as const },
              data: [{ yAxis: 0 }],
              label: { show: false },
            }
          : undefined,
    })),
  };

  return <EChart option={option} height={360} ariaLabel={t("generalization.deltaTitle")} />;
}

export function deltaWelTable(
  generalization: Generalization,
  region: Region,
  t: (k: string) => string,
): TableModel {
  const block = generalization.regions[region];
  const conditions = CONDITIONS.filter((c) => c in block);
  const rows = conditions.flatMap((c) =>
    POLICIES.filter((p) => cell(block, c, p, region)).map((p) => {
      const v = cell(block, c, p, region);
      return {
        display: [
          t(`condition.${c}`),
          t(`policy.${p}`),
          fmtWEL(v.dWEL),
          fmtWEL(v.WEL_mean),
          v.ISR_mean.toFixed(3),
          String(v.n_seeds),
        ],
        csv: [c, p, v.dWEL, v.WEL_mean, v.ISR_mean, v.n_seeds],
      };
    }),
  );
  return {
    columns: [
      { header: t("filters.condition") },
      { header: t("results.baseline") },
      { header: "ΔWEL", numeric: true },
      { header: "WEL", numeric: true },
      { header: "ISR", numeric: true },
      { header: "n", numeric: true },
    ],
    rows: rows.map((r) => r.display),
    csvHeaders: ["condition", "policy", "dWEL", "WEL_mean", "ISR_mean", "n_seeds"],
    csvRows: rows.map((r) => r.csv),
  };
}
