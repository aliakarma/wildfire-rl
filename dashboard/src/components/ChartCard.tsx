import * as echarts from "echarts/core";
import { useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { downloadCsv, triggerDownload } from "../lib/download";
import { usePanelAnchor } from "../lib/usePanelAnchor";
import { token } from "../theme/useTheme";
import { DataTable, type Column } from "./DataTable";

export interface TableModel {
  columns: Column[];
  rows: ReactNode[][];
  /** Plain-text rows for CSV export (render precision, no markup). */
  csvRows: (string | number)[][];
  csvHeaders: string[];
}

interface Props {
  title: string;
  subtitle?: string;
  /** Source artifact path shown in the footnote (§6, provenance per panel). */
  source: string;
  table: TableModel;
  csvName: string;
  children: ReactNode;
  ariaLabel: string;
  extraActions?: ReactNode;
  /** Deep-link id (§5.3): `?panel=<id>` scrolls here; adds a copy-link action. */
  anchorId?: string;
}

/**
 * Chart card (§12): title, protocol subtitle, chart, source footnote, and the chart
 * actions — the mandatory relief-rule pair "View as table" + "Download CSV" (§7.3, §8.4)
 * plus theme-aware "Download PNG". The PNG export resolves the ECharts instance from the
 * rendered DOM (ECharts stamps its init node with `_echarts_instance_`), so individual
 * chart components need no extra plumbing.
 */
export function ChartCard({
  title,
  subtitle,
  source,
  table,
  csvName,
  children,
  ariaLabel,
  extraActions,
  anchorId,
}: Props) {
  const { t } = useTranslation();
  const [asTable, setAsTable] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const anchor = usePanelAnchor<HTMLElement>(anchorId);

  const downloadPng = () => {
    const el = bodyRef.current?.querySelector<HTMLElement>("[_echarts_instance_]");
    const chart = el ? echarts.getInstanceByDom(el) : undefined;
    if (!chart) return;
    triggerDownload(
      chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: token("--surface-1") }),
      csvName.replace(/\.csv$/, ".png"),
    );
  };

  return (
    <figure
      className="card"
      style={{ margin: 0 }}
      aria-label={ariaLabel}
      id={anchorId}
      ref={anchor.ref as React.RefObject<HTMLElement>}
    >
      <div className="chart-card-head">
        <div className="titles">
          <h3 className="card-title">{title}</h3>
          {subtitle && <p className="card-subtitle">{subtitle}</p>}
        </div>
        <div className="chart-actions">
          {extraActions}
          {anchorId && (
            <button onClick={anchor.copyLink}>
              {anchor.copied ? t("actions.copied") : t("actions.copyLink")}
            </button>
          )}
          <button aria-pressed={asTable} onClick={() => setAsTable((v) => !v)}>
            {asTable ? t("actions.viewChart") : t("actions.viewTable")}
          </button>
          <button onClick={() => downloadCsv(csvName, table.csvHeaders, table.csvRows)}>
            {t("actions.downloadCsv")}
          </button>
          {!asTable && <button onClick={downloadPng}>{t("actions.downloadPng")}</button>}
        </div>
      </div>
      <div ref={bodyRef}>
        {asTable ? <DataTable columns={table.columns} rows={table.rows} caption={title} /> : children}
      </div>
      <figcaption className="card-footnote">
        {t("common.sourceArtifact", { path: "" })}
        <code>{source}</code>
      </figcaption>
    </figure>
  );
}
