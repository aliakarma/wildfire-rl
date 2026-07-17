import { useTranslation } from "react-i18next";

import type { TableModel } from "../components/ChartCard";
import type { Region, TrainCurves } from "../lib/data";
import { fmt, fmtWEL } from "../lib/format";
import { methodColor } from "../theme/echartsTheme";
import { token } from "../theme/useTheme";
import { EChart, type EChartsOption } from "./EChart";

const METHODS = ["mappo", "commnet", "hiercomm_heur"];

interface Props {
  curves: TrainCurves;
  region: Region;
  showSeeds: boolean;
}

interface AxisTooltipParam {
  seriesName: string;
  dataIndex: number;
  value: [number, number];
  marker: string;
}

/**
 * Training curves (§6.3): WEL per training episode, median across the 5 seeds per method,
 * optional per-seed spaghetti at 30% opacity, crosshair tooltip with all series (§8.4).
 * The x-axis is the episode number — seeds finish different episode counts within the
 * 100k-step budget, so cross-seed aggregation is aligned by episode, with the mean
 * cumulative env-steps shown in the tooltip for context.
 */
export function TrainingCurvesChart({ curves, region, showSeeds }: Props) {
  const { t } = useTranslation();
  const data = curves.regions[region];
  const textSecondary = token("--text-secondary");
  const border = token("--border");

  const series: object[] = [];
  for (const m of METHODS) {
    const c = data[m];
    const color = methodColor(m);
    if (showSeeds) {
      for (const [seed, values] of Object.entries(c.seeds)) {
        series.push({
          type: "line",
          name: `${t(`policy.${m}`)} · ${seed}`,
          data: c.episode.map((e, i) => [e, values[i]]),
          lineStyle: { width: 1, color, opacity: 0.3 },
          itemStyle: { color },
          showSymbol: false,
          silent: true,
          emphasis: { disabled: true },
          tooltip: { show: false },
          legendHoverLink: false,
        });
      }
    }
    series.push({
      type: "line",
      name: t(`policy.${m}`),
      data: c.episode.map((e, i) => [e, c.median[i]]),
      lineStyle: { width: 2, color },
      itemStyle: { color },
      showSymbol: false,
      endLabel: {
        show: true,
        formatter: () => t(`policy.${m}`),
        color,
        fontSize: 12,
        distance: 6,
      },
      z: 5,
    });
  }

  const stepsRef = data[METHODS[0]];

  const option: EChartsOption = {
    grid: { left: 8, right: 130, top: 16, bottom: 32, containLabel: true },
    legend: {
      show: true,
      data: METHODS.map((m) => t(`policy.${m}`)),
      textStyle: { color: textSecondary, fontSize: 12 },
      top: 0,
      left: 0,
      icon: "roundRect",
      itemWidth: 12,
      itemHeight: 3,
    },
    xAxis: {
      type: "value",
      name: t("results.episodeAxis"),
      nameLocation: "middle",
      nameGap: 26,
      nameTextStyle: { color: textSecondary, fontSize: 12 },
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { color: textSecondary, fontSize: 12, formatter: (v: number) => fmt(v) },
      max: "dataMax",
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
      axisPointer: { type: "cross", label: { show: false } },
      backgroundColor: token("--surface-1"),
      borderColor: border,
      textStyle: { color: token("--text-primary"), fontSize: 12 },
      formatter: (params: AxisTooltipParam[]) => {
        if (!params.length) return "";
        const i = params[0].dataIndex;
        const lines = [
          `${t("results.episodeAxis")} ${fmt(params[0].value[0])} · ${t("results.envSteps")} ${fmt(
            stepsRef.env_steps_mean[i] ?? 0,
          )}`,
        ];
        for (const p of params) {
          lines.push(`${p.marker} ${p.seriesName}: ${fmtWEL(p.value[1])}`);
        }
        return lines.join("<br/>");
      },
    },
    series,
  };

  return <EChart option={option} height={360} ariaLabel={t("results.trainingCurves")} />;
}

export function trainingCurvesTable(
  curves: TrainCurves,
  region: Region,
  t: (k: string) => string,
): TableModel {
  const data = curves.regions[region];
  const episodes = data[METHODS[0]].episode;
  const stepsMean = data[METHODS[0]].env_steps_mean;
  // Table view samples every ~10th point to stay readable; CSV carries all points.
  const stride = Math.max(1, Math.floor(episodes.length / 40));
  const rows: (string | number)[][] = [];
  const csvRows: (string | number)[][] = [];
  episodes.forEach((e, i) => {
    csvRows.push([e, stepsMean[i], ...METHODS.map((m) => data[m].median[i])]);
    if (i % stride === 0) {
      rows.push([fmt(e), fmt(stepsMean[i]), ...METHODS.map((m) => fmtWEL(data[m].median[i]))]);
    }
  });
  return {
    columns: [
      { header: t("results.episodeAxis"), numeric: true },
      { header: t("results.envSteps"), numeric: true },
      ...METHODS.map((m) => ({ header: t(`policy.${m}`), numeric: true })),
    ],
    rows,
    csvHeaders: ["episode", "env_steps_mean", ...METHODS.map((m) => `${m}_median_WEL`)],
    csvRows,
  };
}
