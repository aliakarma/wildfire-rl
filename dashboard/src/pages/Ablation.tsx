import { Trans, useTranslation } from "react-i18next";

import { AblationDumbbell, ablationTable } from "../charts/AblationDumbbell";
import { Callout } from "../components/Callout";
import { ChartCard } from "../components/ChartCard";
import { DataTable } from "../components/DataTable";
import { LoadError, Skeleton } from "../components/Loading";
import { RegionSwitch } from "../components/RegionSwitch";
import { useData, type Ablations, type Region } from "../lib/data";
import { fmtP } from "../lib/format";
import { useRegionChoice } from "../lib/urlState";
import { usePanelAnchor } from "../lib/usePanelAnchor";

function NegativeResultPanel({ ablations }: { ablations: Ablations }) {
  const { t } = useTranslation();
  const anchor = usePanelAnchor<HTMLDivElement>("negative-result");
  // The p-values are interpolated from the frozen artifact — never hand-typed (§2.1).
  const pSaudi = ablations.regions.saudi.wo_comms.vs_full_WEL?.p;
  const pCal = ablations.regions.california.wo_comms.vs_full_WEL?.p;
  if (pSaudi === undefined || pCal === undefined) return null;
  return (
    <div id="negative-result" ref={anchor.ref}>
      <Callout title={t("ablation.negativeTitle")}>
        <p style={{ margin: 0 }}>
          <Trans
            i18nKey="ablation.negativeBody"
            values={{ pSaudi: fmtP(pSaudi), pCalifornia: fmtP(pCal) }}
            components={{ bdi: <bdi dir="ltr" /> }}
          />{" "}
          <button
            onClick={anchor.copyLink}
            style={{
              fontSize: 12,
              color: "var(--text-secondary)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-chip)",
              padding: "1px 6px",
            }}
          >
            {anchor.copied ? t("actions.copied") : t("actions.copyLink")}
          </button>
        </p>
      </Callout>
    </div>
  );
}

function RegionPanels({ region, ablations }: { region: Region; ablations: Ablations }) {
  const { t } = useTranslation();
  const table = ablationTable(ablations, region, t);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-6)" }}>
      <ChartCard
        title={`${t("ablation.contribution")} — ${t(`region.${region}`)}`}
        subtitle={t("ablation.contributionSubtitle")}
        source="results/wildfire_phase4/ablation_summary.json"
        table={table}
        csvName={`ablation_${region}.csv`}
        ariaLabel={`${t("ablation.contribution")}, ${t(`region.${region}`)}`}
        anchorId={`contribution-${region}`}
      >
        <AblationDumbbell ablations={ablations} region={region} />
      </ChartCard>

      <div className="card">
        <h3 className="card-title" style={{ marginBlockEnd: "var(--sp-3)" }}>
          {t("ablation.tableTitle")} — {t(`region.${region}`)}
        </h3>
        <DataTable caption={t("ablation.tableTitle")} columns={table.columns} rows={table.rows} />
        <p className="card-footnote">
          † {t("ablation.deterministicNote")} ·{" "}
          <code>results/wildfire_phase4/ablation_summary.json</code>
        </p>
      </div>
    </div>
  );
}

export default function Ablation() {
  const { t } = useTranslation();
  const [choice] = useRegionChoice();
  const regions: Region[] = choice === "both" ? ["saudi", "california"] : [choice];
  const { data, error, retry } = useData<Ablations>("ablations.json");

  return (
    <>
      <RegionSwitch />
      <div className="page">
        <h1>{t("ablation.title")}</h1>
        <p className="page-summary">{t("ablation.summary")}</p>
        {error ? (
          <LoadError retry={retry} />
        ) : !data ? (
          <Skeleton height={640} />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-6)" }}>
            <NegativeResultPanel ablations={data} />
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  regions.length === 2 ? "repeat(auto-fit, minmax(min(560px, 100%), 1fr))" : "1fr",
                gap: "var(--sp-6)",
              }}
            >
              {regions.map((region) => (
                <RegionPanels key={region} region={region} ablations={data} />
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
