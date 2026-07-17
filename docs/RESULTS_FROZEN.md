# Frozen main results (Phase 3)

**Status: FROZEN — 2026-07-10.** These are the canonical main-results numbers for the paper.

| | |
|---|---|
| Results dir | `wildfire_phase3_multiseed/` (30 checkpoints, curves, summary, table, logs) |
| Git commit | `567945f` (branch `additional`) |
| Manifest fingerprint | `da4a381dc6ab43e5cb2a01b60fd5c8f32f2b111587fa38b0277b100fc35586dd` |
| Protocol | 5 **training** seeds `{42,1042,2042,3042,4042}`; eval = 5 held-out seed groups × 15 episodes |
| Determinism | seeded env-reset pipeline; proven **bit-identical on CPU** |

Verify integrity at any time: `python scripts/freeze_results.py wildfire_phase3_multiseed` → the
`fingerprint_sha256` must match the value above.

> **Provenance note:** `FREEZE.json` records `git_dirty: true` — the freeze was taken from a
> working tree with uncommitted changes at `567945f`. The manifest **fingerprint** (a SHA-256
> over every frozen file), not the commit hash, is therefore the authoritative reference for
> the frozen content. The ablation/robustness (`wildfire_phase4/`) and generalization/transfer
> (`wildfire_phase6/`) directories carry their own `FREEZE.json` fingerprints, pinned by
> `scripts/build_dashboard_data.py`.

## Reproduction command

```bash
python scripts/run_phase3.py \
  --methods noop,value_first,greedy_risk,local_reactive,mappo,commnet,hiercomm_heur \
  --regions saudi,california --train-seeds 42,1042,2042,3042,4042 --parallel 6 \
  --train-steps 100000 --episodes 15 --out wildfire_phase3_multiseed
```

Regime `default` (documented in `src/wildfire_marl/env/regimes.py`): Saudi wind×0.6 / FFMC 90,
California wind×0.5 / FFMC 90; `treat_radius=1`, threat ignition, `ros_cv=0.1`,
`steps_per_action=30`. Training reset seed = `seed·100000 + episode`; eval reset seed =
`group + 100000 + episode` (disjoint streams, no leakage).

## Main results (WEL ↓, mean ± std over 5 training seeds; heuristics deterministic)

| Method | Saudi WEL | Saudi ISR | California WEL | California ISR |
|---|---|---|---|---|
| No-Op | 27.00 | 0.286 | 34.00 | 0.000 |
| Value-First | 18.84 | 0.545 | 28.40 | 0.156 |
| Greedy-Risk | 24.72 | 0.358 | 28.40 | 0.156 |
| Local Reactive | 11.91 | 0.720 | 10.23 | 0.702 |
| Flat MARL (MAPPO) | 9.77 ± 5.10 | 0.692 | 14.17 ± 2.89 | 0.624 |
| CommNet | 12.44 ± 4.32 | 0.682 | 19.58 ± 7.77 | 0.398 |
| **HierComm (ours)** | **7.02 ± 2.36** | **0.809** | **5.88 ± 1.04** | **0.832** |

## Significance — HierComm (ours) vs baselines (WEL; Welch for learned, one-sample for heuristics)

**California — beats every baseline significantly:** MAPPO p=0.0018 (d=−3.81), CommNet p=0.016
(d=−2.47), Local Reactive p=0.0007, Value-First/Greedy/No-Op p<1e-4.

**Saudi — beats every baseline except MAPPO:** CommNet p=0.048, Local Reactive p=0.010,
Value-First p=4e-4, Greedy p=1e-4, No-Op p<1e-4; **vs MAPPO p=0.32 (n.s.)**.

## Headline claims these numbers support (honest framing)

1. HierComm is the **best-in-class** method in both regions (lowest WEL, highest ISR).
2. HierComm is **by far the most reliable** — lowest across-seed variance (±2.36 / ±1.04) vs
   MAPPO (bimodal on Saudi: per-seed `[6.0,14.9,6.6,15.7,5.6]`) and CommNet (±7.77 on
   California). It is the *only* method reliably best across both regions.
3. On California the win is **statistically decisive**; on Saudi it significantly beats all
   baselines except MAPPO, which it ties on the mean while being far less seed-sensitive.

Do **not** use "dominates / 3–5×" language — that was an artifact of a single non-reproducible
draw (see the reproducibility audit).

## Not in this frozen table (deferred to appendix/ablation)

- **QMIX** — degenerate (Saudi = No-Op; California partial). Report as a failed baseline.
- **HierComm (learned commander)** — the fully-learned-`StrategicController` variant; underperforms
  the value-heuristic commander. Report as an ablation that justifies the value dispatcher.

Both can be added by including `qmix,hiercomm` in the sweep's `--methods` (idempotent; ~20 extra
jobs). Until then, cite them at single seed from the earlier deterministic run with a footnote.
