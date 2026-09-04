import { useTranslation } from "react-i18next";

import { DifficultySweep, robustnessTable } from "../charts/DifficultySweep";
import { Callout } from "../components/Callout";
import { ChartCard } from "../components/ChartCard";
import { LoadError, Skeleton } from "../components/Loading";
import { RegionSwitch } from "../components/RegionSwitch";
import { useData, type Region, type Robustness as RobustnessData } from "../lib/data";
import { useRegionChoice } from "../lib/urlState";

/** Count the region × difficulty cells where HierComm has the lowest WEL — computed from
 * the artifact at render time, never hand-typed (guide §2.1). */
function lowestWelCells(data: RobustnessData): { hc: number; total: number } {
  let total = 0;
  let hc = 0;
  for (const region of Object.values(data.regions)) {
    for (const cell of Object.values(region)) {
      total += 1;
      const lowest = Object.entries(cell).reduce((a, b) =>
        b[1].WEL_mean < a[1].WEL_mean ? b : a,
      );
      if (lowest[0] === "hiercomm_heur") hc += 1;
    }
  }
  return { hc, total };
}

export default function Robustness() {
  const { t } = useTranslation();
  const [choice] = useRegionChoice();
  const regions: Region[] = choice === "both" ? ["saudi", "california"] : [choice];
  const { data, error, retry } = useData<RobustnessData>("robustness.json");

  return (
    <>
      <RegionSwitch />
      <div className="page">
        <h1>{t("robustness.title")}</h1>
        <p className="page-summary">{t("robustness.summary")}</p>
        {error ? (
          <LoadError retry={retry} />
        ) : !data ? (
          <Skeleton height={480} />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-6)" }}>
            <Callout>
              <strong>{t("robustness.callout", lowestWelCells(data))}</strong>
              <p style={{ margin: "var(--sp-1) 0 0" }}>{t("robustness.anchor")}</p>
            </Callout>
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  regions.length === 2 ? "repeat(auto-fit, minmax(min(560px, 100%), 1fr))" : "1fr",
                gap: "var(--sp-6)",
              }}
            >
              {regions.map((region) => (
                <ChartCard
                  key={region}
                  title={`${t("robustness.sweep")} — ${t(`region.${region}`)}`}
                  subtitle={t("metric.wel.hint")}
                  source="wildfire_phase4_extended/robustness_summary.json"
                  table={robustnessTable(data, region, t)}
                  csvName={`robustness_${region}.csv`}
                  ariaLabel={`${t("robustness.sweep")}, ${t(`region.${region}`)}`}
                  anchorId={`sweep-${region}`}
                >
                  <DifficultySweep robustness={data} region={region} />
                </ChartCard>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
