import { useTranslation } from "react-i18next";

import type { TableModel } from "../components/ChartCard";
import type { Transfer } from "../lib/data";
import { fmtWEL } from "../lib/format";
import { token } from "../theme/useTheme";
import { EChart, type EChartsOption } from "./EChart";

const CELLS = ["saudi->saudi", "saudi->california", "california->saudi", "california->california"];

interface Props {
  transfer: Transfer;
  policy: string;
}

/**
 * 2×2 transfer heatmap (§6.6): rows = trained-on, columns = evaluated-on; single-hue
 * sequential blue ramp with every cell annotated — color is never the only encoding (§7.4).
 */
export function TransferHeatmap({ transfer, policy }: Props) {
  const { t } = useTranslation();
  const matrix = transfer.policies[policy].matrix;
  const textSecondary = token("--text-secondary");

  const regions = ["saudi", "california"];
  const labels = [t("region.saudiShort"), t("region.californiaShort")];
  const values = CELLS.map((c) => matrix[c].WEL_mean);
  const min = Math.min(...values);
  const max = Math.max(...values);

  const data = CELLS.map((c) => {
    const [from, to] = c.split("->");
    return [regions.indexOf(to), regions.indexOf(from), matrix[c].WEL_mean];
  });

  const option: EChartsOption = {
    grid: { left: 8, right: 16, top: 8, bottom: 48, containLabel: true },
    xAxis: {
      type: "category",
      data: labels,
      name: t("transfer.evaluatedOn"),
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: textSecondary, fontSize: 12 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: textSecondary, fontSize: 12 },
      splitArea: { show: false },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      name: t("transfer.trainedOn"),
      nameLocation: "middle",
      nameGap: 68,
      nameTextStyle: { color: textSecondary, fontSize: 12 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: textSecondary, fontSize: 12 },
    },
    visualMap: {
      show: false,
      min,
      max,
      inRange: { color: [token("--seq-lo"), token("--seq-hi")] },
    },
    tooltip: {
      trigger: "item",
      backgroundColor: token("--surface-1"),
      borderColor: token("--border"),
      textStyle: { color: token("--text-primary"), fontSize: 12 },
      formatter: ({ value }: { value: [number, number, number] }) =>
        `${t("transfer.trainedOn")}: ${labels[value[1]]}<br/>${t("transfer.evaluatedOn")}: ${
          labels[value[0]]
        }<br/>WEL: ${fmtWEL(value[2])}`,
    },
    series: [
      {
        type: "heatmap",
        data,
        label: {
          show: true,
          formatter: ({ value }: { value: [number, number, number] }) => fmtWEL(value[2]),
          fontSize: 14,
          fontFamily: "IBM Plex Mono, monospace",
        },
        itemStyle: { borderColor: token("--surface-1"), borderWidth: 2 },
        emphasis: { itemStyle: { opacity: 0.85 } },
      },
    ],
  };

  return (
    <EChart
      option={option}
      height={240}
      ariaLabel={`${t("transfer.matrixTitle")} — ${transfer.policies[policy].label}`}
    />
  );
}

export function transferTable(transfer: Transfer, t: (k: string) => string): TableModel {
  const rows: { display: (string | number)[]; csv: (string | number)[] }[] = [];
  for (const [key, pol] of Object.entries(transfer.policies)) {
    for (const [dir, d] of Object.entries(pol.directions)) {
      rows.push({
        display: [
          t(`policy.${key}`),
          dir,
          fmtWEL(d.native_WEL),
          fmtWEL(d.transfer_WEL),
          d.native_ISR.toFixed(3),
          d.transfer_ISR.toFixed(3),
          d.TRS_ISR.toFixed(3),
        ],
        csv: [key, dir, d.native_WEL, d.transfer_WEL, d.native_ISR, d.transfer_ISR, d.TRS_ISR],
      });
    }
  }
  return {
    columns: [
      { header: t("results.baseline") },
      { header: "→" },
      { header: `${t("transfer.native")} WEL`, numeric: true },
      { header: `${t("transfer.transferred")} WEL`, numeric: true },
      { header: `${t("transfer.native")} ISR`, numeric: true },
      { header: `${t("transfer.transferred")} ISR`, numeric: true },
      { header: "TRS", numeric: true },
    ],
    rows: rows.map((r) => r.display),
    csvHeaders: ["policy", "direction", "native_WEL", "transfer_WEL", "native_ISR", "transfer_ISR", "TRS_ISR"],
    csvRows: rows.map((r) => r.csv),
  };
}
