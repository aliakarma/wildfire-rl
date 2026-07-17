import { useTranslation } from "react-i18next";

import { Callout } from "../components/Callout";
import { DataTable } from "../components/DataTable";
import {
  dataUrl,
  useData,
  type BenchmarkData,
  type MainResults,
  type MediaIndex,
} from "../lib/data";
import { fmtWEL } from "../lib/format";

/** Regime parameters shown in the table, in display order (values come from
 * benchmark.json, which the build script extracts from regimes.py — no drift). */
const REGIME_PARAMS = [
  "wind_scale",
  "ffmc",
  "ignition_dist",
  "ignition_mode",
  "treat_radius",
  "steps_per_action",
  "ros_cv",
  "max_steps",
] as const;

const SWEEP = ["easy", "medium", "hard"] as const;

function fmtParam(v: unknown): string {
  if (Array.isArray(v)) return `${v[0]}–${v[1]}`;
  return String(v);
}

/** One cell: the Saudi value, plus the California value when it differs. */
function regimeCell(regimes: BenchmarkData["regimes"], regime: string, param: string): string {
  const saudi = fmtParam(regimes.saudi[regime][param]);
  const cal = fmtParam(regimes.california[regime][param]);
  return saudi === cal ? saudi : `${saudi} / ${cal}`;
}

const GLOSSARY = ["wel", "isr", "ce", "dwel", "trs"] as const;

export default function Benchmark() {
  const { t } = useTranslation();
  const media = useData<MediaIndex>("media.json");
  const bench = useData<BenchmarkData>("benchmark.json");
  const results = useData<MainResults>("main_results.json");

  const noopWel = (region: "saudi" | "california") =>
    results.data ? fmtWEL(results.data.regions[region].policies.noop.WEL.mean) : "…";

  const snapshot = (region: string) =>
    media.data?.items.find((i) => i.kind === "rollout" && i.region === region && i.policy === "noop")
      ?.poster;

  return (
    <div className="page">
      <h1>{t("benchmark.title")}</h1>
      <p className="page-summary">{t("benchmark.summary")}</p>

      <section className="section">
        <h2>{t("benchmark.regionCards")}</h2>
        <div className="grid-2">
          {(["saudi", "california"] as const).map((region) => (
            <div className="card" key={region}>
              <h3 className="card-title" style={{ marginBlockEnd: "var(--sp-2)" }}>
                {t(`region.${region}`)}
              </h3>
              {snapshot(region) && (
                <div
                  style={{
                    background: "#f6f5f2",
                    borderRadius: "var(--radius-chip)",
                    overflow: "hidden",
                    marginBlockEnd: "var(--sp-3)",
                  }}
                >
                  <img
                    src={dataUrl(snapshot(region)!)}
                    alt={t(`region.${region}`)}
                    loading="lazy"
                  />
                </div>
              )}
              <p style={{ fontSize: 14, color: "var(--text-secondary)", margin: 0 }}>
                {t(`benchmark.${region}Desc`, { wel: noopWel(region) })}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <h2>{t("benchmark.regime")}</h2>
        <p className="page-summary">{t("benchmark.regimeIntro")}</p>
        {bench.data && (
          <>
            <DataTable
              caption={t("benchmark.regime")}
              columns={[
                { header: t("benchmark.regimeParam") },
                { header: t("difficulty.easy"), numeric: true },
                { header: t("difficulty.medium"), numeric: true },
                { header: t("difficulty.hard"), numeric: true },
              ]}
              rows={REGIME_PARAMS.map((param) => [
                <code key="p">{param}</code>,
                ...SWEEP.map((regime) => regimeCell(bench.data!.regimes, regime, param)),
              ])}
            />
            <p className="card-footnote">
              {t("benchmark.regimeCellNote")} · <code>{bench.data.source}</code>
            </p>
          </>
        )}
      </section>

      <section className="section">
        <h2>{t("benchmark.pipeline")}</h2>
        <ol
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "var(--sp-3)",
            listStyle: "none",
            margin: 0,
            padding: 0,
            counterReset: "step",
          }}
        >
          {(["ndvi", "era5", "srtm", "firms", "crit"] as const).map((step, i) => (
            <li
              key={step}
              className="card"
              style={{
                padding: "var(--sp-3) var(--sp-4)",
                fontSize: 14,
                display: "flex",
                alignItems: "center",
                gap: "var(--sp-2)",
              }}
            >
              <span
                aria-hidden
                style={{
                  color: "var(--accent)",
                  fontWeight: 650,
                  fontFeatureSettings: '"tnum"',
                }}
              >
                {i + 1}
              </span>
              <bdi>{t(`benchmark.pipelineSteps.${step}`)}</bdi>
            </li>
          ))}
        </ol>
      </section>

      <section className="section">
        <h2>{t("benchmark.glossary")}</h2>
        <DataTable
          caption={t("benchmark.glossary")}
          columns={[{ header: "" }, { header: "" }]}
          rows={GLOSSARY.map((m) => [
            <strong key="n">{t(`metric.${m}.name`)}</strong>,
            t(`metric.${m}.hint`),
          ])}
        />
      </section>

      <section className="section">
        <h2>{t("benchmark.boundary")}</h2>
        <div className="grid-2">
          <Callout title="Cell2Fire">{t("benchmark.boundaryPhysics")}</Callout>
          <Callout title="Wrapper">{t("benchmark.boundaryWrapper")}</Callout>
        </div>
      </section>
    </div>
  );
}
