import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

import { useTheme } from "../theme/useTheme";

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  HeatmapChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export type EChartsOption = echarts.EChartsCoreOption;

interface Props {
  option: EChartsOption;
  height: number;
  /** Summary sentence for screen readers (§13); the table view is the canonical alternative. */
  ariaLabel: string;
}

/**
 * Thin ECharts wrapper: tree-shaken imports, resize-aware, and re-rendered from the
 * current design tokens whenever the theme toggles (§9.2 — no reload).
 */
export function EChart({ option, height, ariaLabel }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts>();
  const { theme } = useTheme();

  useEffect(() => {
    if (!ref.current) return;
    chart.current = echarts.init(ref.current);
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(ref.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
    };
  }, []);

  useEffect(() => {
    // `theme` in the dependency list forces a token re-read after a toggle.
    chart.current?.setOption(option, { notMerge: true });
  }, [option, theme]);

  return <div ref={ref} style={{ height, width: "100%" }} role="img" aria-label={ariaLabel} />;
}
