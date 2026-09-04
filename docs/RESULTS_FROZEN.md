# Frozen main results (Phase 3)

**Status: FROZEN — 2026-07-10.** These are the canonical main-results numbers for the paper.

| | |
|---|---|
| Results dir | `results/wildfire_phase3_multiseed/` (30 checkpoints, curves, summary, table, logs) |
| Git commit | `567945f` (branch `additional`) |
| Manifest fingerprint | `da4a381dc6ab43e5cb2a01b60fd5c8f32f2b111587fa38b0277b100fc35586dd` |
| Protocol | 5 **training** seeds `{42,1042,2042,3042,4042}`; eval = 5 held-out seed groups × 15 episodes |
| Determinism | seeded env-reset pipeline; proven **bit-identical on CPU** |

Verify integrity at any time: `python scripts/freeze_results.py results/wildfire_phase3_multiseed` → the
`fingerprint_sha256` must match the value above.

> **Provenance note:** `FREEZE.json` records `git_dirty: true` — the freeze was taken from a
> working tree with uncommitted changes at `567945f`. The manifest **fingerprint** (a SHA-256
> over every frozen file), not the commit hash, is therefore the authoritative reference for
> the frozen content. The ablation/robustness (`results/wildfire_phase4/`) and generalization/transfer
> (`results/wildfire_phase6/`) directories carry their own `FREEZE.json` fingerprints, pinned by
> `scripts/build_dashboard_data.py`.
>
> **Known irreproducibility (Greedy-Risk only, 2026-07-19 audit):** re-running the frozen
> protocol at HEAD reproduces every per-episode record bit-exactly **except** parts of the
> Greedy-Risk stream — a pre-freeze dirty-tree behavioral delta confined to that policy's
> collision-prone trajectories (divergences are single 3×3 treatment patches; burned-cell
> deltas of exactly 9). Aggregates: Saudi Greedy-Risk reproduces exactly (WEL 24.72,
> ISR 0.3581); California reproduces to WEL 28.32 / ISR 0.1578 vs the frozen 28.40 / 0.1556
> (Δ ≤ 0.08 WEL, no ranking or conclusion affected). All other policies reproduce exactly.
> Dashboard replays for Greedy-Risk therefore use episodes that verify bit-exactly against
> the frozen per-episode CSV (Saudi: group 42 ep 1; California: group 42 ep 8).

## Reproduction command

```bash
python scripts/run_phase3.py \
  --methods noop,value_first,greedy_risk,local_reactive,mappo,commnet,hiercomm_heur \
  --regions saudi,california --train-seeds 42,1042,2042,3042,4042 --parallel 6 \
  --train-steps 100000 --episodes 15 --out results/wildfire_phase3_multiseed
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

## Extended difficulty sweep (2026-07-19)

`results/wildfire_phase4_extended/` (FREEZE fingerprint `cada146a7eacd3d6…`) extends the Phase-4
difficulty sweep to **all seven policies** and supersedes `results/wildfire_phase4/` as the source of
the paper's difficulty table and the dashboard's Robustness page (ablation data still comes
from `results/wildfire_phase4/`). Protocol: easy/hard computed at 10 episodes per eval-seed group;
the **medium column equals the `default` regime and reuses the main 15-episode evaluation**
(`scripts/postprocess_extended_robustness.py` normalizes it from `phase3_summary.json` so no
column mixes budgets — the same convention the original 4-policy artifact used implicitly).
Headline: Local Reactive is best in both easy cells (Saudi 3.06, California 0.00); HierComm
is best in all four medium/hard cells (largest margins on hard).

## Not in this frozen table (deferred to appendix/ablation)

- **QMIX** — degenerate (Saudi = No-Op; California partial). Report as a failed baseline.
- **HierComm (learned commander)** — ~~single-seed deferred~~ **completed 2026-07-19** at the
  full 5-seed protocol: `results/wildfire_phase3_hiercomm_learned/` (FREEZE fingerprint
  `925624778a321458…`, 10 checkpoints + curves + summary). Result: statistically
  indistinguishable from the value-aware rule on Saudi (WEL 6.02 ± 2.04 vs 7.02 ± 2.36,
  Welch p=0.49) and significantly **worse** on California (8.20 ± 1.63 vs 5.88 ± 1.04,
  p=0.032, d=1.70) — it never significantly improves, justifying the deployed rule-based
  dispatcher. Reproduce: `python scripts/run_phase3.py --methods hiercomm --regions
  saudi,california --train-seeds 42,1042,2042,3042,4042 --parallel 3 --train-steps 100000
  --episodes 15 --out results/wildfire_phase3_hiercomm_learned`.

QMIX can still be added by including `qmix` in the sweep's `--methods` (idempotent); until
then, cite it at single seed from the earlier deterministic run with a footnote.
