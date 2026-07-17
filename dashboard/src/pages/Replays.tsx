import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Callout } from "../components/Callout";
import { LoadError, Skeleton } from "../components/Loading";
import { MediaFrame } from "../components/MediaFrame";
import { RegionSwitch } from "../components/RegionSwitch";
import { useData, type MediaIndex, type MediaItem, type Region } from "../lib/data";
import { useRegionChoice } from "../lib/urlState";
import { ReplayViewer } from "../replay/ReplayViewer";
import type { ReplayIndex } from "../replay/decode";

const KINDS = ["comparison", "rollout", "transfer_native", "transfer_cross"] as const;

function caption(item: MediaItem, t: (k: string, o?: Record<string, unknown>) => string): string {
  const region = t(`region.${item.region}`);
  const kind = t(`replays.kind.${item.kind}`);
  const policy = item.policy === "all" ? t("policy.all") : t(`policy.${item.policy}`);
  const direction = item.direction
    ? ` · ${item.direction
        .split("->")
        .map((r) => t(`region.${r}Short`))
        .join(" → ")}`
    : "";
  return `${policy} — ${region} · ${kind}${direction}`;
}

export default function Replays() {
  const { t } = useTranslation();
  const [choice] = useRegionChoice();
  const { data, error, retry } = useData<MediaIndex>("media.json");
  const replayIndex = useData<ReplayIndex>("replays/index.json");
  const [kindFilter, setKindFilter] = useState<string>("all");
  const [policyFilter, setPolicyFilter] = useState<string>("all");

  const regions: Region[] = choice === "both" ? ["saudi", "california"] : [choice as Region];
  // The interactive viewer shows one region at a time (compare mode is the second panel).
  const viewerRegion: Region = choice === "both" ? "saudi" : (choice as Region);

  const items = (data?.items ?? []).filter(
    (i) =>
      regions.includes(i.region) &&
      (kindFilter === "all" || i.kind === kindFilter) &&
      (policyFilter === "all" || i.policy === policyFilter),
  );

  const policies = Array.from(new Set((data?.items ?? []).map((i) => i.policy))).filter(
    (p) => p !== "all",
  );

  return (
    <>
      <RegionSwitch />
      <div className="page">
        <h1>{t("replays.title")}</h1>
        <p className="page-summary">{t("replays.summary")}</p>

        <Callout>{t("replays.honesty")}</Callout>

        {replayIndex.data && replayIndex.data.replays.length > 0 && (
          <div style={{ margin: "var(--sp-6) 0" }}>
            <ReplayViewer region={viewerRegion} entries={replayIndex.data.replays} />
          </div>
        )}

        <h2 style={{ marginBlockStart: "var(--sp-8)" }}>{t("replays.gallery")}</h2>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "var(--sp-4)",
            margin: "var(--sp-6) 0",
            alignItems: "center",
          }}
        >
          <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>{t("filters.kind")}</span>
          <div className="segmented" role="group" aria-label={t("filters.kind")}>
            <button aria-pressed={kindFilter === "all"} onClick={() => setKindFilter("all")}>
              {t("policy.all")}
            </button>
            {KINDS.map((k) => (
              <button key={k} aria-pressed={kindFilter === k} onClick={() => setKindFilter(k)}>
                {t(`replays.kind.${k}`)}
              </button>
            ))}
          </div>
          <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            {t("results.baseline")}
          </span>
          <div className="segmented" role="group" aria-label={t("results.baseline")}>
            <button aria-pressed={policyFilter === "all"} onClick={() => setPolicyFilter("all")}>
              {t("policy.all")}
            </button>
            {policies.map((p) => (
              <button key={p} aria-pressed={policyFilter === p} onClick={() => setPolicyFilter(p)}>
                {p === "all" ? t("policy.all") : t(`policy.${p}`)}
              </button>
            ))}
          </div>
        </div>

        {error ? (
          <LoadError retry={retry} />
        ) : !data ? (
          <Skeleton height={480} />
        ) : items.length === 0 ? (
          <div className="error-card">
            <p>{t("common.noData")}</p>
            <button
              onClick={() => {
                setKindFilter("all");
                setPolicyFilter("all");
              }}
              style={{
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-control)",
                padding: "4px 12px",
              }}
            >
              {t("filters.reset")}
            </button>
          </div>
        ) : (
          <div className="grid-2">
            {items.map((item) => (
              <MediaFrame key={item.id} item={item} caption={caption(item, t)} />
            ))}
          </div>
        )}

        <p className="card-footnote" style={{ marginBlockStart: "var(--sp-6)" }}>
          <code>figures/wildfire_phase5_gifs · figures/wildfire_phase6_gifs</code>
        </p>
      </div>
    </>
  );
}
