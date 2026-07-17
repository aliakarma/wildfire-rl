import { useEffect, useRef } from "react";

import type { Region } from "../lib/data";
import { useBasemap, type BasemapRegion } from "./basemap";
import type { DecodedReplay } from "./decode";

export interface OverlayState {
  assets: boolean;
  targets: boolean;
  treated: boolean;
  trails: boolean;
}

interface Props {
  replay: DecodedReplay;
  t: number;
  overlays: OverlayState;
  /** Localized summary for screen readers (§13). */
  ariaLabel: string;
}

/* ------------------------------------------------------------------ */
/* Mirrored from wildfire_marl/viz/geo_renderer.py — the renderer behind the Phase-5/6
/* GIFs. Any change there must be reflected here, or the viewer and the GIFs will show the
/* same rollout differently.                                                             */
/* ------------------------------------------------------------------ */

/** ``_FIRE_CMAP`` anchors: new fire → burnt out, evenly spaced. */
const FIRE_RGB: [number, number, number][] = [
  [255, 235, 59],
  [255, 152, 0],
  [244, 67, 54],
  [183, 28, 28],
  [74, 0, 0],
];
const AGENT_COLORS = ["#00B0FF", "#00E676", "#E040FB"];
const AGENT_LABELS = ["A1", "A2", "A3"];
const TARGET_COLOR = "#AB47BC";
const COMM_COLOR = "rgba(171, 71, 188, ";
const COMM_POLICIES = new Set(["hiercomm_heur", "hiercomm", "commnet"]);
const HIERARCHICAL = new Set(["hiercomm_heur", "hiercomm"]);

/** ``_THEMES["paper"]`` — the theme the committed GIFs are rendered with. */
const FIRE_ALPHA = 0.72;
const GLOW = 1.0;
const GRID_ALPHA = 0.1;
const WASH_ALPHA = 0.06;

const TRAIL_LENGTH = 10;

/** Asset marker outlines (``_HOUSE_MKR`` / ``_DERRICK_MKR``), in marker units, y up. */
const HOUSE: [number, number][] = [
  [0, 0.55], [-0.42, 0], [-0.3, 0], [-0.3, -0.45],
  [0.3, -0.45], [0.3, 0], [0.42, 0],
];
const DERRICK: [number, number][] = [
  [0, 0.55], [-0.28, -0.45], [-0.1, -0.45], [-0.04, 0.12],
  [0.04, 0.12], [0.1, -0.45], [0.28, -0.45],
];

/**
 * ``LinearSegmentedColormap.from_list`` over evenly spaced anchors — fire colour by
 * normalized age (0 = just ignited, 1 = burnt out).
 */
function fireColor(na: number): string {
  const x = Math.min(Math.max(na, 0), 1) * (FIRE_RGB.length - 1);
  const i = Math.min(Math.floor(x), FIRE_RGB.length - 2);
  const f = x - i;
  const [r, g, b] = [0, 1, 2].map((k) =>
    Math.round(FIRE_RGB[i][k] + (FIRE_RGB[i + 1][k] - FIRE_RGB[i][k]) * f),
  );
  return `${r},${g},${b}`;
}

/** ``_rounded_patch``: a cell-sized rounded rect, expanded about the cell centre. */
function roundedCell(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  halfW: number,
  halfH: number,
  expand: number,
  rpadFrac: number,
): void {
  const rpad = halfW * rpadFrac;
  const w = halfW * 2 * expand;
  const h = halfH * 2 * expand;
  const x = cx - w / 2;
  const y = cy - h / 2;
  const r = Math.min(rpad, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Geometry for a region with no exported basemap: a plain uniform grid. */
function fallbackGeom(w: number, h: number): BasemapRegion {
  return {
    image: "",
    width: w,
    height: h,
    grid: w,
    displayName: "",
    assetLabel: "",
    assetColor: "#FF6D00",
    cellX: Array.from({ length: w }, (_, c) => (c + 0.5) / w),
    cellY: Array.from({ length: h }, (_, r) => (r + 0.5) / h),
    halfW: 0.5 / w,
    halfH: 0.5 / h,
    bounds: { lat: [0, 0], lon: [0, 0] },
    scaleBarFrac: 0,
  };
}

/**
 * Replay grid renderer (§11.4): the simulation drawn over the region's real basemap, in the
 * same visual language as the Phase-5/6 GIFs — fire coloured by age, hatched firebreaks,
 * per-region asset icons, and numbered agents with trails. The grid is a map: it never
 * mirrors under RTL, and fire vs firebreak is carried by lightness and shape (hatch, icons),
 * never hue alone.
 */
export function ReplayCanvas({ replay, t, overlays, ariaLabel }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const basemap = useBasemap(replay.meta.region as Region);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const { h, w } = replay;
    const geom = basemap?.meta ?? fallbackGeom(w, h);
    const dpr = devicePixelRatio;
    const maxCss = Math.min(640, canvas.parentElement?.clientWidth ?? 640);

    canvas.width = Math.round(maxCss * dpr);
    canvas.height = Math.round((canvas.width * geom.height) / geom.width);
    canvas.style.width = `${canvas.width / dpr}px`;
    canvas.style.height = `${canvas.height / dpr}px`;
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const px = (c: number) => geom.cellX[c] * W;
    const py = (r: number) => geom.cellY[r] * H;
    const halfW = geom.halfW * W;
    const halfH = geom.halfH * H;
    // The GIFs are drawn in points at 150 dpi with ~33.75 px cells; deriving a point unit
    // from the cell size reproduces their exact proportions at any canvas resolution.
    const pt = (halfW * 2) / 16.2;

    ctx.clearRect(0, 0, W, H);
    if (basemap) {
      ctx.drawImage(basemap.img, 0, 0, W, H);
    } else {
      ctx.fillStyle = "#2d2d3d"; // GeoRenderer's basemap-unavailable fill.
      ctx.fillRect(0, 0, W, H);
    }

    // Dark wash — lifts terrain contrast and pushes back the basemap's labels.
    ctx.fillStyle = `rgba(0,0,0,${WASH_ALPHA})`;
    ctx.fillRect(0, 0, W, H);

    // Grid.
    ctx.strokeStyle = `rgba(255,255,255,${GRID_ALPHA})`;
    ctx.lineWidth = 0.3 * pt;
    ctx.beginPath();
    for (let r = 0; r <= h; r++) {
      const y = r < h ? py(r) - halfH : py(h - 1) + halfH;
      ctx.moveTo(px(0) - halfW, y);
      ctx.lineTo(px(w - 1) + halfW, y);
    }
    for (let c = 0; c <= w; c++) {
      const x = c < w ? px(c) - halfW : px(w - 1) + halfW;
      ctx.moveTo(x, py(0) - halfH);
      ctx.lineTo(x, py(h - 1) + halfH);
    }
    ctx.stroke();

    // Cell state at step t. A cell can be treated after burning (or overrun after being
    // treated), so the later transition wins — mirroring the sign of fire_state.
    const burning: number[] = [];
    const firebreak: number[] = [];
    for (let i = 0; i < replay.igniteAt.length; i++) {
      const ig = replay.igniteAt[i];
      const tr = replay.treatAt[i];
      const isIg = ig !== -1 && ig <= t;
      const isTr = tr !== -1 && tr <= t;
      if (isIg && (!isTr || ig > tr)) burning.push(i);
      else if (isTr && (!isIg || tr > ig)) firebreak.push(i);
    }

    const maxAge = Math.max(t, 1);
    const normAge = (i: number) =>
      Math.min((t - replay.igniteAt[i]) / Math.max(maxAge * 0.5, 1), 1);

    // Fire glow.
    for (const i of burning) {
      const na = normAge(i);
      const a = Math.max((0.14 - 0.08 * na) * GLOW, 0.02);
      ctx.fillStyle = `rgba(255,209,0,${a})`;
      roundedCell(ctx, px(i % w), py(Math.floor(i / w)), halfW, halfH, 1.35, 0.2);
      ctx.fill();
    }

    // Firebreak glow.
    if (overlays.treated) {
      ctx.fillStyle = "rgba(77,191,255,0.08)";
      for (const i of firebreak) {
        roundedCell(ctx, px(i % w), py(Math.floor(i / w)), halfW, halfH, 1.2, 0.18);
        ctx.fill();
      }
    }

    // Fire body — colour and opacity carry age.
    ctx.lineWidth = 0.5 * pt;
    for (const i of burning) {
      const na = normAge(i);
      const rgb = fireColor(na);
      ctx.fillStyle = `rgba(${rgb},${FIRE_ALPHA - 0.18 * na})`;
      ctx.strokeStyle = `rgba(${rgb},${0.4 - 0.2 * na})`;
      roundedCell(ctx, px(i % w), py(Math.floor(i / w)), halfW, halfH, 1.04, 0.14);
      ctx.fill();
      ctx.stroke();
    }

    // Firebreaks: translucent cyan, then a 45° hatch — the shape cue that keeps them
    // distinguishable from fire without relying on hue.
    if (overlays.treated) {
      ctx.lineWidth = 0.7 * pt;
      ctx.fillStyle = "rgba(140,235,255,0.28)";
      ctx.strokeStyle = "rgba(140,242,255,0.75)";
      for (const i of firebreak) {
        roundedCell(ctx, px(i % w), py(Math.floor(i / w)), halfW, halfH, 1.0, 0.12);
        ctx.fill();
        ctx.stroke();
      }
      ctx.strokeStyle = "rgba(140,242,255,0.40)";
      ctx.lineWidth = 0.4 * pt;
      ctx.beginPath();
      for (const i of firebreak) {
        const x0 = px(i % w) - halfW;
        const y0 = py(Math.floor(i / w)) - halfH;
        const cw = halfW * 2;
        const ch = halfH * 2;
        for (let k = 1; k < 5; k++) {
          const f = k / 5;
          ctx.moveTo(x0 + cw * f, y0 + ch);
          ctx.lineTo(x0, y0 + ch * (1 - f));
          ctx.moveTo(x0 + cw, y0 + ch * f);
          ctx.lineTo(x0 + cw * (1 - f), y0);
        }
      }
      ctx.stroke();
    }

    // Fresh-ignition hot core.
    for (const i of burning) {
      const age = t - replay.igniteAt[i];
      if (age > 2) continue;
      ctx.fillStyle = `rgba(255,255,217,${0.3 * (1 - age / 3)})`;
      roundedCell(ctx, px(i % w), py(Math.floor(i / w)), halfW, halfH, 0.55, 0.25);
      ctx.fill();
    }

    const burnedSet = new Set(burning);
    const breakSet = new Set(firebreak);

    // Assets: a region-specific icon (house = WUI community, derrick = infrastructure),
    // ringed green once defended, greyed with a red outline once burned.
    if (overlays.assets) {
      const shape = replay.meta.region === "saudi" ? DERRICK : HOUSE;
      const size = Math.sqrt(120) * pt;
      for (const a of replay.assets) {
        const i = a.y * w + a.x;
        const cx = px(a.x);
        const cy = py(a.y);
        const burned = burnedSet.has(i);
        const defended = breakSet.has(i);

        if (defended) {
          ctx.strokeStyle = "rgba(0,230,118,0.31)";
          ctx.lineWidth = 1.8 * pt;
          ctx.beginPath();
          ctx.arc(cx, cy, halfW * 0.75, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.globalAlpha = burned ? 0.5 : 0.95;
        ctx.fillStyle = burned ? "#666666" : "#FFFFFF";
        ctx.strokeStyle = burned ? "#FF0000" : defended ? "#00E676" : geom.assetColor;
        ctx.lineWidth = 1.4 * pt;
        ctx.beginPath();
        shape.forEach(([mx, my], k) => {
          const X = cx + mx * size;
          const Y = cy - my * size; // marker space is y-up.
          if (k === 0) ctx.moveTo(X, Y);
          else ctx.lineTo(X, Y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    const frame = replay.frames[Math.min(t, replay.frames.length - 1)];
    const agents = frame.agents;

    // Communication links: a pulse travelling along each pairwise channel.
    if (COMM_POLICIES.has(replay.meta.policy) && agents.length > 1) {
      const nPts = 24;
      const pulsePos = (t * 0.18) % 1.0;
      ctx.lineWidth = 1.4 * pt;
      for (let i = 0; i < agents.length; i++) {
        for (let j = i + 1; j < agents.length; j++) {
          const x1 = px(agents[i][1]);
          const y1 = py(agents[i][0]);
          const x2 = px(agents[j][1]);
          const y2 = py(agents[j][0]);
          const dx = x2 - x1;
          const dy = y2 - y1;
          // Perpendicular bow, sign-corrected for the canvas' y-down axis.
          const mx = (x1 + x2) / 2 + dy * 0.1;
          const my = (y1 + y2) / 2 - dx * 0.1;
          const pts: [number, number][] = [];
          for (let k = 0; k <= nPts; k++) {
            const s = k / nPts;
            const u = 1 - s;
            pts.push([
              u * u * x1 + 2 * u * s * mx + s * s * x2,
              u * u * y1 + 2 * u * s * my + s * s * y2,
            ]);
          }
          for (let k = 0; k < pts.length - 1; k++) {
            let dist = Math.abs((k + 0.5) / nPts - pulsePos);
            if (dist > 0.5) dist = 1 - dist;
            const alpha = 0.12 + 0.45 * Math.max(0, 1 - dist * 5);
            ctx.strokeStyle = `${COMM_COLOR}${alpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(pts[k][0], pts[k][1]);
            ctx.lineTo(pts[k + 1][0], pts[k + 1][1]);
            ctx.stroke();
          }
        }
      }
    }

    // Agent trails: the last 10 positions, widening and brightening toward the present.
    if (overlays.trails) {
      agents.forEach((_, idx) => {
        const trail: [number, number][] = [];
        for (let back = TRAIL_LENGTH - 1; back >= 0; back--) {
          const past = replay.frames[t - back];
          const a = past?.agents[idx];
          if (a) trail.push([px(a[1]), py(a[0])]);
        }
        if (trail.length < 2) return;
        const color = AGENT_COLORS[idx % AGENT_COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineCap = "round";
        for (let k = 0; k < trail.length - 1; k++) {
          const frac = k / Math.max(trail.length - 1, 1);
          ctx.globalAlpha = 0.06 + 0.4 * frac;
          ctx.lineWidth = (0.8 + 1.8 * frac) * pt;
          ctx.beginPath();
          ctx.moveTo(trail[k][0], trail[k][1]);
          ctx.lineTo(trail[k + 1][0], trail[k + 1][1]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.lineCap = "butt";
      });
    }

    // Commander targets (hierarchical policies): dashed assignment line, pulsing ring, X.
    if (overlays.targets && HIERARCHICAL.has(replay.meta.policy)) {
      frame.targets.forEach(([ty, tx], idx) => {
        const agent = agents[idx];
        const cx = px(tx);
        const cy = py(ty);
        if (agent) {
          ctx.strokeStyle = "rgba(171,71,188,0.40)";
          ctx.lineWidth = 0.9 * pt;
          ctx.setLineDash([3 * pt, 3 * pt]);
          ctx.beginPath();
          ctx.moveTo(px(agent[1]), py(agent[0]));
          ctx.lineTo(cx, cy);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        const pulseR = halfW * (0.55 + 0.15 * Math.sin(t * 0.7));
        const pulseA = 0.22 + 0.12 * Math.sin(t * 0.7);
        ctx.fillStyle = "rgba(171,71,188,0.06)";
        ctx.strokeStyle = `rgba(171,71,188,${pulseA.toFixed(3)})`;
        ctx.lineWidth = 1.3 * pt;
        ctx.beginPath();
        ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        const s = Math.sqrt(75) * pt * 0.5;
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 0.7 * pt + 2.2 * pt;
        for (const pass of [0, 1]) {
          if (pass === 1) {
            ctx.strokeStyle = TARGET_COLOR;
            ctx.lineWidth = 2.2 * pt;
          }
          ctx.beginPath();
          ctx.moveTo(cx - s, cy - s);
          ctx.lineTo(cx + s, cy + s);
          ctx.moveTo(cx + s, cy - s);
          ctx.lineTo(cx - s, cy + s);
          ctx.stroke();
        }
      });
    }

    // Agents: a coloured disc with a white ring and its call sign.
    agents.forEach(([ay, ax], idx) => {
      const cx = px(ax);
      const cy = py(ay);
      const color = AGENT_COLORS[idx % AGENT_COLORS.length];
      ctx.fillStyle = color;
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 1.8 * pt;
      // matplotlib scatter s is area in pt²; the 'o' marker diameter is sqrt(s) pt.
      ctx.beginPath();
      ctx.arc(cx, cy, (Math.sqrt(180) / 2) * pt, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#FFFFFF";
      ctx.font = `bold ${7 * pt}px "IBM Plex Mono", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(AGENT_LABELS[idx % AGENT_LABELS.length], cx, cy);
    });

    // Map furniture: 100 km scale bar, north arrow, and the tile provider's attribution
    // (a licence condition of the basemap — the GIFs carry the same string).
    if (basemap) {
      const barLen = geom.scaleBarFrac * W;
      const bx = W * 0.05;
      const by = H - H * 0.04;
      const th = Math.max(1.5, H * 0.005);
      for (let s = 0; s < 4; s++) {
        ctx.fillStyle = s % 2 === 0 ? "#FFFFFF" : "#333333";
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 0.3 * pt;
        ctx.fillRect(bx + (s * barLen) / 4, by - th, barLen / 4, th * 2);
        ctx.strokeRect(bx + (s * barLen) / 4, by - th, barLen / 4, th * 2);
      }
      const label = Math.max(9, 4.5 * pt);
      ctx.font = `bold ${label}px "IBM Plex Mono", monospace`;
      ctx.textBaseline = "top";
      ctx.fillStyle = "#FFFFFF";
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 2;
      for (const [v, off] of [
        ["0", 0],
        ["50", 0.5],
        ["100 km", 1],
      ] as [string, number][]) {
        ctx.textAlign = "center";
        ctx.strokeText(v, bx + barLen * off, by + th * 2);
        ctx.fillText(v, bx + barLen * off, by + th * 2);
      }

      const nx = W - W * 0.05;
      const ny = H * 0.06;
      const ah = H * 0.035;
      const aw = W * 0.011;
      ctx.fillStyle = "#FFFFFF";
      ctx.strokeStyle = "#cccccc";
      ctx.lineWidth = 0.4 * pt;
      ctx.beginPath();
      ctx.moveTo(nx, ny + ah * 0.7);
      ctx.lineTo(nx - aw, ny + ah * 1.4);
      ctx.lineTo(nx + aw, ny + ah * 1.4);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.font = `bold ${7.5 * pt}px "IBM Plex Sans", sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillStyle = "#FFFFFF";
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 2;
      ctx.strokeText("N", nx, ny + ah * 0.62);
      ctx.fillText("N", nx, ny + ah * 0.62);

      const attr = Math.max(8, 4.5 * pt);
      ctx.font = `${attr}px "IBM Plex Sans", sans-serif`;
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      const tw = ctx.measureText(basemap.attribution).width;
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(W - tw - attr * 1.2, H - attr * 1.8, tw + attr * 1.0, attr * 1.6);
      ctx.fillStyle = "#dddddd";
      ctx.fillText(basemap.attribution, W - attr * 0.7, H - attr * 0.3);
    }
  }, [replay, t, overlays, basemap]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      style={{ display: "block", borderRadius: "var(--radius-chip)", maxWidth: "100%" }}
    />
  );
}
