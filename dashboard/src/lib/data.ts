import { useEffect, useState } from "react";

/* ------------------------------------------------------------------ */
/* Typed contracts for the JSON emitted by scripts/build_dashboard_data.py */
/* ------------------------------------------------------------------ */

export type Region = "saudi" | "california";
export type RegionChoice = Region | "both";
export const REGIONS: Region[] = ["saudi", "california"];

export interface Meta {
  fingerprint: string;
  frozen_utc: string;
  freeze_commit: string;
  commit: string;
  built: string;
  protocol: { train_seeds: number[]; episodes: number; unit: string };
}

export interface MetricSummary {
  mean: number;
  std: number;
  ci: [number, number];
  n_train_seeds: number;
  values: number[];
}

export interface PolicyResult {
  learned: boolean;
  WEL: MetricSummary;
  ISR: MetricSummary;
  CE: MetricSummary;
  burned: MetricSummary;
  return: MetricSummary;
}

export interface Comparison {
  t: number;
  p: number;
  d: number;
  test: "welch" | "one-sample";
}

export interface MainResults {
  meta: Meta;
  policy_order: string[];
  regions: Record<
    Region,
    {
      policies: Record<string, PolicyResult>;
      comparisons: Record<"WEL" | "ISR", Record<string, Comparison>>;
    }
  >;
}

export interface TrainCurve {
  /** Aligned by episode index — seeds finish different episode counts (see build script). */
  episode: number[];
  /** Mean cumulative env steps across seeds at each kept episode (tooltip context). */
  env_steps_mean: number[];
  median: number[];
  seeds: Record<string, number[]>;
  rolling_window: number;
  n_episodes: number;
  episodes_per_seed: Record<string, number>;
}

export interface TrainCurves {
  meta: Meta;
  regions: Record<Region, Record<string, TrainCurve>>;
}

export interface AblationVariant {
  label: string;
  WEL_mean: number;
  WEL_std: number;
  WEL_ci: [number, number];
  ISR_mean: number;
  dWEL_vs_noop: number;
  n_seeds: number;
  vs_full_WEL?: { p: number; d: number };
}

export interface Ablations {
  meta: Meta;
  regions: Record<Region, Record<string, AblationVariant>>;
}

export interface Robustness {
  meta: Meta;
  protocol: { regimes: string[] };
  regions: Record<
    Region,
    Record<string, Record<string, { WEL_mean: number; ISR_mean: number }>>
  >;
}

export interface GeneralizationCell {
  WEL_mean: number;
  WEL_ci: [number, number];
  ISR_mean: number;
  dWEL: number;
  n_seeds: number;
}

export interface Generalization {
  meta: Meta;
  regions: Record<Region, Record<string, Record<string, GeneralizationCell>>>;
}

export interface TransferDirection {
  native_ISR: number;
  transfer_ISR: number;
  TRS_ISR: number;
  native_WEL: number;
  transfer_WEL: number;
  WEL_gap: number;
  ISR_degradation_p: number;
  ISR_degradation_d: number;
}

export interface Transfer {
  meta: Meta;
  policies: Record<
    string,
    {
      label: string;
      directions: Record<string, TransferDirection>;
      matrix: Record<string, { WEL_mean: number; WEL_std: number; ISR_mean: number; ISR_std: number }>;
    }
  >;
}

export interface MediaItem {
  id: string;
  kind: "comparison" | "rollout" | "transfer_native" | "transfer_cross";
  region: Region;
  policy: string;
  /** MP4 when available (compress_media.py), else the original GIF. */
  src: string;
  gif_fallback?: string;
  poster?: string;
  bytes: number;
  direction?: string;
}

export interface MediaIndex {
  meta: Meta;
  items: MediaItem[];
}

export interface BenchmarkData {
  meta: Meta;
  source: string;
  regimes: Record<Region, Record<string, Record<string, unknown>>>;
}

export interface Reproducibility {
  meta: Meta;
  seed42_checkpoint_sha256: Record<string, string>;
  n_checkpoints: number;
  reproduction_command: string;
  verify_command: string;
  compute: string;
}

/* ------------------------------------------------------------------ */
/* Loader                                                              */
/* ------------------------------------------------------------------ */

const cache = new Map<string, unknown>();
const inflight = new Map<string, Promise<unknown>>();

export function dataUrl(rel: string): string {
  return `${import.meta.env.BASE_URL}${rel}`;
}

async function fetchJson<T>(name: string): Promise<T> {
  if (cache.has(name)) return cache.get(name) as T;
  if (!inflight.has(name)) {
    const request = fetch(dataUrl(`data/${name}`))
      .then(async (r) => {
        if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
        const json = await r.json();
        cache.set(name, json);
        return json;
      })
      // Always clear the inflight slot — a cached rejection would make retry() a no-op.
      .finally(() => inflight.delete(name));
    inflight.set(name, request);
  }
  return inflight.get(name) as Promise<T>;
}

export interface Loaded<T> {
  data?: T;
  error?: string;
  retry: () => void;
}

/** Fetch one of the build-emitted JSON files, with module-level caching. */
export function useData<T>(name: string): Loaded<T> {
  const [state, setState] = useState<{ data?: T; error?: string; attempt: number }>({
    attempt: 0,
  });

  useEffect(() => {
    let live = true;
    fetchJson<T>(name)
      .then((data) => live && setState((s) => ({ ...s, data, error: undefined })))
      .catch((e) => live && setState((s) => ({ ...s, error: String(e) })));
    return () => {
      live = false;
    };
  }, [name, state.attempt]);

  return {
    data: state.data,
    error: state.error,
    retry: () => setState((s) => ({ attempt: s.attempt + 1 })),
  };
}
