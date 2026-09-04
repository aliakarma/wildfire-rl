import { useState } from "react";
import { useTranslation } from "react-i18next";

import { ChartCard } from "../components/ChartCard";
import { DataTable } from "../components/DataTable";
import { LoadError, Skeleton } from "../components/Loading";
import { MethodChip } from "../components/MethodChip";
import { RegionSwitch } from "../components/RegionSwitch";
import { SigBadge } from "../components/SigBadge";
import { ResultsBars, resultsBarsTable } from "../charts/ResultsBars";
import { TrainingCurvesChart, trainingCurvesTable } from "../charts/TrainingCurves";
import { useData, type MainResults, type Region, type TrainCurves } from "../lib/data";
import { fmt2, fmtP } from "../lib/format";
import { useRegionChoice } from "../lib/urlState";

const BASELINE_ORDER = [
  "proposed_vs_mappo",
  "proposed_vs_commnet",
  "proposed_vs_local_reactive",
  "proposed_vs_value_first",
  "proposed_vs_greedy_risk",
  "proposed_vs_noop",
];

function StatsPanel({ results, region }: { results: MainResults; region: Region }) {
  const { t } = useTranslation();
  const comps = results.regions[region].comparisons.WEL;
  return (
    <div className="card">
      <h3 className="card-title">
        {t("results.statsPanel")} — {t(`region.${region}`)}
      </h3>
      <p className="card-subtitle" style={{ marginBlockEnd: "var(--sp-3)" }}>
        {t("results.statsSubtitle")}
      </p>
      <DataTable
        caption={t("results.statsPanel")}
        columns={[
          { header: t("results.baseline") },
          { header: "t", numeric: true },
          { header: t("stats.pvalue"), numeric: true },
          { header: t("stats.effect"), numeric: true },
          { header: t("stats.test") },
          { header: "" },
        ]}
        rows={BASELINE_ORDER.filter((k) => comps[k]).map((k) => {
          const c = comps[k];
          const policy = k.replace("proposed_vs_", "");
          return [
            <MethodChip key="chip" policy={policy} />,
            fmt2(c.t),
            <bdi key="p" dir="ltr">
              {fmtP(c.p)}
            </bdi>,
            fmt2(c.d),
            c.test === "welch" ? t("stats.welch") : t("stats.oneSample"),
            <SigBadge key="sig" p={c.p} />,
          ];
        })}
      />
      <p className="card-footnote">
        {t("common.sourceArtifact", { path: "" })}
        <code>results/wildfire_phase3_multiseed/phase3_summary.json</code>
      </p>
    </div>
  );
}

function RegionPanels({ region }: { region: Region }) {
  const { t } = useTranslation();
  const results = useData<MainResults>("main_results.json");
  const curves = useData<TrainCurves>("train_curves.json");
  const [showSeeds, setShowSeeds] = useState(false);

  if (results.error) return <LoadError retry={results.retry} />;
  if (!results.data) return <Skeleton height={720} />;
  const r = results.data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-6)" }}>
      <ChartCard
        title={`${t("results.welByPolicy")} — ${t(`region.${region}`)}`}
        subtitle={`${t("metric.wel.hint")} · ${t("common.protocolNote")}`}
        source="results/wildfire_phase3_multiseed/phase3_summary.json"
        table={resultsBarsTable(r, region, "WEL", t)}
        csvName={`wel_by_policy_${region}.csv`}
        ariaLabel={`${t("results.welByPolicy")}, ${t(`region.${region}`)}`}
        anchorId={`wel-${region}`}
      >
        <ResultsBars results={r} region={region} metric="WEL" />
      </ChartCard>

      <ChartCard
        title={`${t("results.isrByPolicy")} — ${t(`region.${region}`)}`}
        subtitle={t("metric.isr.hint")}
        source="results/wildfire_phase3_multiseed/phase3_summary.json"
        table={resultsBarsTable(r, region, "ISR", t)}
        csvName={`isr_by_policy_${region}.csv`}
        ariaLabel={`${t("results.isrByPolicy")}, ${t(`region.${region}`)}`}
        anchorId={`isr-${region}`}
      >
        <ResultsBars results={r} region={region} metric="ISR" />
      </ChartCard>

      <StatsPanel results={r} region={region} />

      {curves.error ? (
        <LoadError retry={curves.retry} />
      ) : !curves.data ? (
        <Skeleton height={420} />
      ) : (
        <ChartCard
          title={`${t("results.trainingCurves")} — ${t(`region.${region}`)}`}
          subtitle={t("results.curvesSubtitle", {
            window: curves.data.regions[region].mappo.rolling_window,
            n: Math.min(
              ...Object.values(curves.data.regions[region]).map((c) => c.n_episodes),
            ),
          })}
          source="results/wildfire_phase3_multiseed/train_curve_checkpoint_*.csv"
          table={trainingCurvesTable(curves.data, region, t)}
          csvName={`train_curves_${region}.csv`}
          ariaLabel={`${t("results.trainingCurves")}, ${t(`region.${region}`)}`}
          anchorId={`curves-${region}`}
          extraActions={
            <button aria-pressed={showSeeds} onClick={() => setShowSeeds((v) => !v)}>
              {t("results.perSeedToggle")}
            </button>
          }
        >
          <TrainingCurvesChart curves={curves.data} region={region} showSeeds={showSeeds} />
        </ChartCard>
      )}
    </div>
  );
}

export default function Results() {
  const { t } = useTranslation();
  const [choice] = useRegionChoice();
  const regions: Region[] = choice === "both" ? ["saudi", "california"] : [choice];

  return (
    <>
      <RegionSwitch />
      <div className="page">
        <h1>{t("results.title")}</h1>
        <p className="page-summary">{t("results.summary")}</p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: regions.length === 2 ? "repeat(auto-fit, minmax(min(560px, 100%), 1fr))" : "1fr",
            gap: "var(--sp-6)",
          }}
        >
          {regions.map((region) => (
            <RegionPanels key={region} region={region} />
          ))}
        </div>
      </div>
    </>
  );
}
