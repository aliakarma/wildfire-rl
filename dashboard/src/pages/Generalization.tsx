import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { DeltaWelBars, deltaWelTable } from "../charts/DeltaWelBars";
import { TransferHeatmap, transferTable } from "../charts/TransferHeatmap";
import { Callout } from "../components/Callout";
import { ChartCard } from "../components/ChartCard";
import { LoadError, Skeleton } from "../components/Loading";
import { RegionSwitch } from "../components/RegionSwitch";
import { StatTile } from "../components/StatTile";
import {
  useData,
  type Generalization as GeneralizationData,
  type Region,
  type Transfer,
} from "../lib/data";
import { fmtISR } from "../lib/format";
import { useRegionChoice } from "../lib/urlState";

function TransferSection({ transfer }: { transfer: Transfer }) {
  const { t } = useTranslation();
  const table = transferTable(transfer, t);
  const hc = transfer.policies.hiercomm_heur.directions;
  const cn = transfer.policies.commnet.directions;
  const cnTrs = [cn["saudi->california"].TRS_ISR, cn["california->saudi"].TRS_ISR];
  const good = Math.max(...cnTrs);
  const bad = Math.min(...cnTrs);

  return (
    <section className="section">
      <h2>{t("transfer.matrixTitle")}</h2>
      <div className="grid-2" style={{ marginBlockEnd: "var(--sp-6)" }}>
        {(["hiercomm_heur", "commnet"] as const).map((policy) => (
          <ChartCard
            key={policy}
            title={`${t(`policy.${policy}`)}`}
            subtitle={t("generalization.matrixSubtitle")}
            source="wildfire_phase6/transfer_summary.json"
            table={table}
            csvName="transfer_matrix.csv"
            ariaLabel={`${t("transfer.matrixTitle")} — ${t(`policy.${policy}`)}`}
            anchorId={`transfer-${policy}`}
          >
            <TransferHeatmap transfer={transfer} policy={policy} />
          </ChartCard>
        ))}
      </div>
      <div className="grid-4" style={{ marginBlockEnd: "var(--sp-6)" }}>
        <StatTile
          label={`HierComm TRS — ${t("region.saudiShort")} → ${t("region.californiaShort")}`}
          value={fmtISR(hc["saudi->california"].TRS_ISR)}
        />
        <StatTile
          label={`HierComm TRS — ${t("region.californiaShort")} → ${t("region.saudiShort")}`}
          value={fmtISR(hc["california->saudi"].TRS_ISR)}
        />
        <StatTile
          label={`CommNet TRS — ${t("region.saudiShort")} → ${t("region.californiaShort")}`}
          value={fmtISR(cn["saudi->california"].TRS_ISR)}
        />
        <StatTile
          label={`CommNet TRS — ${t("region.californiaShort")} → ${t("region.saudiShort")}`}
          value={fmtISR(cn["california->saudi"].TRS_ISR)}
        />
      </div>
      <Callout>
        {t("generalization.asymmetry", { good: fmtISR(good), bad: fmtISR(bad) })}{" "}
        <Link to="/replays">{t("generalization.transferReplays")}</Link>
      </Callout>
    </section>
  );
}

export default function Generalization() {
  const { t } = useTranslation();
  const [choice] = useRegionChoice();
  const regions: Region[] = choice === "both" ? ["saudi", "california"] : [choice];
  const gen = useData<GeneralizationData>("generalization.json");
  const transfer = useData<Transfer>("transfer.json");

  return (
    <>
      <RegionSwitch />
      <div className="page">
        <h1>{t("generalization.title")}</h1>
        <p className="page-summary">{t("generalization.summary")}</p>

        {gen.error ? (
          <LoadError retry={gen.retry} />
        ) : !gen.data ? (
          <Skeleton height={420} />
        ) : (
          <section className="section">
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  regions.length === 2 ? "repeat(auto-fit, minmax(min(560px, 100%), 1fr))" : "1fr",
                gap: "var(--sp-6)",
                marginBlockEnd: "var(--sp-6)",
              }}
            >
              {regions.map((region) => (
                <ChartCard
                  key={region}
                  title={`${t("generalization.deltaTitle")} — ${t(`region.${region}`)}`}
                  subtitle={t("generalization.deltaSubtitle")}
                  source="wildfire_phase6/generalization_summary.json"
                  table={deltaWelTable(gen.data!, region, t)}
                  csvName={`generalization_${region}.csv`}
                  ariaLabel={`${t("generalization.deltaTitle")}, ${t(`region.${region}`)}`}
                  anchorId={`dwel-${region}`}
                >
                  <DeltaWelBars generalization={gen.data!} region={region} />
                </ChartCard>
              ))}
            </div>
            <Callout>{t("generalization.honesty")}</Callout>
          </section>
        )}

        {transfer.error ? (
          <LoadError retry={transfer.retry} />
        ) : !transfer.data ? (
          <Skeleton height={360} />
        ) : (
          <TransferSection transfer={transfer.data} />
        )}
      </div>
    </>
  );
}
