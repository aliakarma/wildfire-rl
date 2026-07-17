import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { useData, type Region } from "../lib/data";
import { fmt, fmtISR, fmtWEL } from "../lib/format";
import { MethodChip } from "../components/MethodChip";
import { Skeleton } from "../components/Loading";
import { useBasemap } from "./basemap";
import { decodeReplay, treatedCount, type Replay, type ReplayIndexEntry } from "./decode";
import { ReplayCanvas, type OverlayState } from "./ReplayCanvas";

const BASE_STEP_MS = 180;
const SPEEDS = [0.5, 1, 2, 4];

function entryKey(e: ReplayIndexEntry): string {
  return e.file;
}

function Pane({
  file,
  entry,
  t,
  overlays,
}: {
  file: string;
  entry: ReplayIndexEntry;
  t: number;
  overlays: OverlayState;
}) {
  const { t: tr } = useTranslation();
  const { data } = useData<Replay>(`replays/${file}`);
  const decoded = useMemo(() => (data ? decodeReplay(data) : undefined), [data]);
  if (!decoded) return <Skeleton height={360} />;

  const ti = Math.min(t, decoded.frames.length - 1);
  const frame = decoded.frames[ti];
  const label =
    entry.kind === "transfer"
      ? `${tr(`policy.${entry.policy}`)} · ${tr("replay.transferFrom", {
          region: tr(`region.${entry.ckpt_region}Short`),
        })}`
      : tr(`policy.${entry.policy}`);

  return (
    <div className="replay-pane">
      <div className="pane-head">
        <MethodChip policy={entry.policy} labelOverride={label} />
      </div>
      <ReplayCanvas
        replay={decoded}
        t={ti}
        overlays={overlays}
        ariaLabel={`${label} — ${tr("replay.step")} ${fmt(ti)} / ${fmt(decoded.meta.steps)}`}
      />
      <div className="replay-meters tnum" aria-live="off">
        <span>
          WEL <strong>{fmtWEL(frame.wel)}</strong>
        </span>
        <span>
          ISR <strong>{fmtISR(frame.isr)}</strong>
        </span>
        <span>
          {tr("replay.burned")} <strong>{fmt(frame.burned)}</strong>
        </span>
        <span>
          {tr("overlay.treated")} <strong>{fmt(treatedCount(decoded, ti))}</strong>
        </span>
      </div>
    </div>
  );
}

/**
 * Tier-2 interactive replay viewer (§11.3): synchronized compare mode — two canvases
 * locked to one scrubber — playback controls, live meters, overlay toggles, and full
 * keyboard operability (Space = play/pause, ←/→ = step, Home = restart).
 */
export function ReplayViewer({ region, entries }: { region: Region; entries: ReplayIndexEntry[] }) {
  const { t: tr } = useTranslation();
  const basemap = useBasemap(region);
  const assetColor = basemap?.meta.assetColor ?? "#FF6D00";
  const regionEntries = entries.filter((e) => e.region === region);
  const defaultMain =
    regionEntries.find((e) => e.policy === "hiercomm_heur" && e.kind === "native") ??
    regionEntries[0];
  const defaultCmp = regionEntries.find((e) => e.policy === "noop") ?? null;

  const [mainKey, setMainKey] = useState(defaultMain ? entryKey(defaultMain) : "");
  const [cmpKey, setCmpKey] = useState(defaultCmp ? entryKey(defaultCmp) : "");
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [overlays, setOverlays] = useState<OverlayState>({
    assets: true,
    targets: true,
    treated: true,
    trails: false,
  });
  const stageRef = useRef<HTMLDivElement>(null);

  const main = regionEntries.find((e) => entryKey(e) === mainKey);
  const cmp = regionEntries.find((e) => entryKey(e) === cmpKey);

  // Region switches invalidate the current selection.
  useEffect(() => {
    if (!main) {
      setMainKey(defaultMain ? entryKey(defaultMain) : "");
      setCmpKey(defaultCmp ? entryKey(defaultCmp) : "");
      setT(0);
      setPlaying(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region]);

  const maxT = Math.max(main?.steps ?? 0, cmp?.steps ?? 0);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setT((prev) => {
        if (prev >= maxT) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, BASE_STEP_MS / speed);
    return () => clearInterval(id);
  }, [playing, speed, maxT]);

  if (!main) return null;

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === " ") {
      e.preventDefault();
      setPlaying((p) => !p);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      setT((v) => Math.min(maxT, v + 1));
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      setT((v) => Math.max(0, v - 1));
    } else if (e.key === "Home") {
      e.preventDefault();
      setT(0);
    }
  };

  const select = (value: string, which: "main" | "cmp") => {
    setT(0);
    setPlaying(false);
    if (which === "main") setMainKey(value);
    else setCmpKey(value);
  };

  const optionLabel = (e: ReplayIndexEntry) =>
    e.kind === "transfer"
      ? `${tr(`policy.${e.policy}`)} — ${tr("replay.transferFrom", {
          region: tr(`region.${e.ckpt_region}Short`),
        })}`
      : tr(`policy.${e.policy}`);

  return (
    <section className="card replay-viewer" aria-label={tr("replay.interactiveTitle")}>
      <div className="chart-card-head">
        <div className="titles">
          <h3 className="card-title">{tr("replay.interactiveTitle")}</h3>
          <p className="card-subtitle">
            {tr("replay.verifiedCaption", { seed: String(main.episode_seed) })}
          </p>
        </div>
      </div>

      <div className="replay-pickers">
        <label>
          {tr("replay.policy")}
          <select value={mainKey} onChange={(e) => select(e.target.value, "main")}>
            {regionEntries.map((e) => (
              <option key={entryKey(e)} value={entryKey(e)}>
                {optionLabel(e)}
              </option>
            ))}
          </select>
        </label>
        <label>
          {tr("replay.compare")}
          <select value={cmpKey} onChange={(e) => select(e.target.value, "cmp")}>
            <option value="">{tr("replay.compareNone")}</option>
            {regionEntries
              .filter((e) => entryKey(e) !== mainKey)
              .map((e) => (
                <option key={entryKey(e)} value={entryKey(e)}>
                  {optionLabel(e)}
                </option>
              ))}
          </select>
        </label>
      </div>

      <div
        ref={stageRef}
        className="replay-stage"
        tabIndex={0}
        onKeyDown={onKeyDown}
        aria-label={tr("replay.interactiveTitle")}
      >
        <div className={cmp ? "grid-2" : undefined}>
          <Pane file={main.file} entry={main} t={t} overlays={overlays} />
          {cmp && <Pane file={cmp.file} entry={cmp} t={t} overlays={overlays} />}
        </div>
      </div>

      <div className="replay-controls">
        <button onClick={() => setT(0)} aria-label={tr("replay.reset")} title={tr("replay.reset")}>
          ⏮
        </button>
        <button
          onClick={() => setT((v) => Math.max(0, v - 1))}
          aria-label={`${tr("replay.step")} −1`}
        >
          ‹
        </button>
        <button
          className="play-btn"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? tr("replay.pause") : tr("replay.play")}
          aria-pressed={playing}
        >
          {playing ? "⏸" : "▶"}
        </button>
        <button
          onClick={() => setT((v) => Math.min(maxT, v + 1))}
          aria-label={`${tr("replay.step")} +1`}
        >
          ›
        </button>
        <input
          type="range"
          min={0}
          max={maxT}
          value={Math.min(t, maxT)}
          onChange={(e) => {
            setPlaying(false);
            setT(Number(e.target.value));
          }}
          aria-label={tr("replay.step")}
          dir="ltr"
        />
        <span className="tnum readout">
          <bdi dir="ltr">
            {fmt(Math.min(t, maxT))} / {fmt(maxT)}
          </bdi>
        </span>
        <label className="speed">
          {tr("replay.speed")}
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {fmt(s)}×
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="replay-overlays">
        <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>
          {tr("replay.overlays")}
        </span>
        {(
          [
            ["assets", tr("overlay.assets")],
            ["targets", tr("overlay.targets")],
            ["treated", tr("overlay.treated")],
            ["trails", tr("replay.trails")],
          ] as const
        ).map(([key, label]) => (
          <label key={key}>
            <input
              type="checkbox"
              checked={overlays[key]}
              onChange={() => setOverlays((o) => ({ ...o, [key]: !o[key] }))}
            />
            {label}
          </label>
        ))}
      </div>

      {/* Swatches mirror the canvas, which mirrors the GIF renderer (geo_renderer.py). */}
      <div className="legend-chips" aria-hidden>
        <span className="chip">
          <i style={{ background: "#FFEB3B" }} /> {tr("overlay.fire")}
        </span>
        <span className="chip">
          <i style={{ background: "#B71C1C" }} /> {tr("overlay.burnedOut")}
        </span>
        <span className="chip">
          <i style={{ background: "#8DEDFF" }} /> {tr("overlay.treated")}
        </span>
        <span className="chip">
          <i style={{ background: "#ffffff", border: `2px solid ${assetColor}` }} />{" "}
          {tr("overlay.assets")}
        </span>
        <span className="chip">
          <i style={{ background: "#00B0FF", border: "1.5px solid #ffffff" }} />{" "}
          {tr("overlay.agents")}
        </span>
        <span className="chip">
          <i style={{ background: "#AB47BC" }} /> {tr("overlay.targets")}
        </span>
      </div>
    </section>
  );
}
