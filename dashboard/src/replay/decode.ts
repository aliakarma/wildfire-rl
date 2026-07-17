/**
 * Replay frame data (emitted by scripts/export_replay_frames.py, §11.2).
 * Bitmaps arrive as sparse cell-index deltas (y * W + x); this module reconstructs
 * per-cell ignition/treatment timelines so any step can be rendered in O(cells).
 */

export interface ReplayFinal {
  WEL: number;
  ISR: number;
  CE: number;
  burned: number;
  return: number;
}

export interface ReplayMeta {
  region: string;
  policy: string;
  ckpt_region: string;
  train_seed: number | null;
  episode: number;
  episode_seed: number;
  grid: [number, number];
  steps: number;
  final: ReplayFinal;
}

export interface ReplayAsset {
  y: number;
  x: number;
  type: number;
  value: number;
}

export interface ReplayFrame {
  t: number;
  fire: number[];
  treated: number[];
  agents: [number, number][];
  targets: [number, number][];
  wel: number;
  isr: number;
  burned: number;
}

export interface Replay {
  meta: ReplayMeta;
  static: { fuel: number[]; criticality: number[]; assets: ReplayAsset[] };
  frames: ReplayFrame[];
}

export interface ReplayIndexEntry {
  file: string;
  kind: "native" | "transfer";
  region: string;
  policy: string;
  ckpt_region: string;
  episode_seed: number;
  steps: number;
  final: ReplayFinal;
}

export interface ReplayIndex {
  replays: ReplayIndexEntry[];
  skipped_unverified: { region: string; policy: string; kind: string; reason: string }[];
}

export interface DecodedReplay {
  meta: ReplayMeta;
  h: number;
  w: number;
  fuel: Uint8Array;
  /** Normalized criticality per cell (0–1). */
  crit: Float32Array;
  assets: ReplayAsset[];
  /** Frame index at which each cell ignited / was treated; -1 = never. */
  igniteAt: Int16Array;
  treatAt: Int16Array;
  frames: ReplayFrame[];
}

export function decodeReplay(replay: Replay): DecodedReplay {
  const [h, w] = replay.meta.grid;
  const n = h * w;
  const fuel = new Uint8Array(n);
  for (const i of replay.static.fuel) fuel[i] = 1;
  const crit = new Float32Array(n);
  replay.static.criticality.forEach((v, i) => {
    crit[i] = v;
  });
  const igniteAt = new Int16Array(n).fill(-1);
  const treatAt = new Int16Array(n).fill(-1);
  for (const frame of replay.frames) {
    for (const i of frame.fire) if (igniteAt[i] === -1) igniteAt[i] = frame.t;
    for (const i of frame.treated) if (treatAt[i] === -1) treatAt[i] = frame.t;
  }
  return {
    meta: replay.meta,
    h,
    w,
    fuel,
    crit,
    assets: replay.static.assets,
    igniteAt,
    treatAt,
    frames: replay.frames,
  };
}

/** Count of treated cells up to and including step t. */
export function treatedCount(replay: DecodedReplay, t: number): number {
  let count = 0;
  for (let i = 0; i < replay.treatAt.length; i++) {
    if (replay.treatAt[i] !== -1 && replay.treatAt[i] <= t) count++;
  }
  return count;
}
