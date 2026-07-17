import { useTranslation } from "react-i18next";
import { Link, useSearchParams } from "react-router-dom";

import { Callout } from "../components/Callout";
import { LoadError, Skeleton } from "../components/Loading";
import { MediaFrame } from "../components/MediaFrame";
import { StatTile } from "../components/StatTile";
import { useData, type MainResults, type MediaIndex } from "../lib/data";
import { fmtISR, fmtP, fmtWEL } from "../lib/format";

function WhatWorksCard() {
  const { t } = useTranslation();
  const rows = [
    { key: "hierarchy", significant: true },
    { key: "tactical", significant: true },
    { key: "finetune", significant: true },
    { key: "comms", significant: false },
  ];
  return (
    <div className="card">
      <h3 className="card-title" style={{ marginBlockEnd: "var(--sp-3)" }}>
        {t("overview.whatWorks.title")}
      </h3>
      <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: "var(--sp-2)" }}>
        {rows.map((r) => (
          <li key={r.key} style={{ display: "flex", gap: "var(--sp-2)", alignItems: "baseline" }}>
            <span aria-hidden style={{ color: "var(--text-secondary)", flex: "none" }}>
              {r.significant ? "✓" : "◦"}
            </span>
            <span>
              {t(`overview.whatWorks.${r.key}`)}
              <span
                style={{ display: "block", fontSize: 13, color: "var(--text-muted)" }}
              >
                {r.significant ? (
                  t("overview.whatWorks.significant")
                ) : (
                  <Link to="/ablation" style={{ color: "var(--text-muted)" }}>
                    {t("overview.whatWorks.commsNote")}
                  </Link>
                )}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Overview() {
  const { t } = useTranslation();
  const results = useData<MainResults>("main_results.json");
  const media = useData<MediaIndex>("media.json");
  const [params] = useSearchParams();
  const withParams = (to: string) => {
    const q = params.toString();
    return q ? `${to}?${q}` : to;
  };

  if (results.error) {
    return (
      <div className="page">
        <LoadError retry={results.retry} />
      </div>
    );
  }
  if (!results.data) {
    return (
      <div className="page">
        <Skeleton height={560} />
      </div>
    );
  }

  const r = results.data;
  const saudi = r.regions.saudi.policies;
  const cal = r.regions.california.policies;
  const pMappo = r.regions.saudi.comparisons.WEL.proposed_vs_mappo.p;
  const comparison = media.data?.items.find(
    (i) => i.kind === "comparison" && i.region === "california",
  );

  return (
    <div className="page">
      <section className="section" style={{ marginBlockEnd: "var(--sp-8)" }}>
        <h1 style={{ fontSize: "clamp(28px, 4vw, 40px)", maxWidth: "26ch" }}>
          {t("overview.title")}
        </h1>
        <p className="page-summary" style={{ fontSize: 17 }}>
          {t("overview.subtitle")}
        </p>
      </section>

      <section className="section grid-4" style={{ marginBlockEnd: "var(--sp-8)" }}>
        <StatTile
          label={t("overview.tiles.welSaudi")}
          value={`${fmtWEL(saudi.hiercomm_heur.WEL.mean)} ±${fmtWEL(saudi.hiercomm_heur.WEL.std)}`}
          context={`${t("common.vsNoop", { value: fmtWEL(saudi.noop.WEL.mean) })} · ${t("common.lowerBetter")}`}
        />
        <StatTile
          label={t("overview.tiles.welCalifornia")}
          value={`${fmtWEL(cal.hiercomm_heur.WEL.mean)} ±${fmtWEL(cal.hiercomm_heur.WEL.std)}`}
          context={`${t("common.vsNoop", { value: fmtWEL(cal.noop.WEL.mean) })} · ${t("common.lowerBetter")}`}
        />
        <StatTile
          label={t("overview.tiles.isrBoth")}
          value={`${fmtISR(saudi.hiercomm_heur.ISR.mean)} / ${fmtISR(cal.hiercomm_heur.ISR.mean)}`}
          context={`${t("overview.tiles.isrContext")} · ${t("common.higherBetter")}`}
        />
        <StatTile
          label={t("overview.tiles.protocol")}
          value={t("overview.tiles.protocolValue")}
          context={t("overview.tiles.protocolContext")}
        />
      </section>

      <section className="section grid-2">
        {comparison ? (
          <MediaFrame
            item={comparison}
            caption={t("overview.comparisonCaption", { region: t("region.california") })}
          />
        ) : media.error || media.data ? (
          // media.json failed, or loaded without a comparison item — degrade quietly.
          <div className="error-card">{t("replays.mediaUnavailable")}</div>
        ) : (
          <Skeleton height={320} />
        )}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-6)" }}>
          <WhatWorksCard />
          <Callout>{t("overview.honesty", { p: fmtP(pMappo) })}</Callout>
        </div>
      </section>

      <section style={{ display: "flex", flexWrap: "wrap", gap: "var(--sp-3)" }}>
        <Link className="nav-link" style={{ border: "1px solid var(--border)" }} to={withParams("/results")}>
          {t("actions.exploreResults")} →
        </Link>
        <Link className="nav-link" style={{ border: "1px solid var(--border)" }} to={withParams("/replays")}>
          {t("actions.watchReplays")} →
        </Link>
        <Link className="nav-link" style={{ border: "1px solid var(--border)" }} to={withParams("/about")}>
          {t("actions.readPaper")} →
        </Link>
      </section>
    </div>
  );
}
