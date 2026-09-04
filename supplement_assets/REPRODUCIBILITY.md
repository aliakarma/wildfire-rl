# Reproducibility: claim → command → expected number

Every quantitative claim in the paper, the command that produces it, and the frozen artifact
that already contains it. Ordered by cost: the first two tiers need no GPU and no simulator.

---

## Tier 1 — verify the numbers (seconds, no GPU, no simulator build)

```bash
pip install -r requirements.txt
python scripts/verify_supplement.py
```

This re-derives the paper's tables from the frozen **per-episode** CSVs and diffs them against
the values printed in `docs/RESULTS_FROZEN.md`. It checks 42 assertions and exits non-zero if
any fails. Expected output ends with:

```
42 passed, 0 failed, 0 skipped
VERIFICATION PASSED — every claim checked re-derives from the frozen artifacts.
```

What it checks:

| Group | Assertions | Source of truth |
|---|---|---|
| Main table: WEL and ISR for 7 policies × 2 regions | 14 | `wildfire_phase3_multiseed/phase3_raw.csv` |
| Published summary agrees with raw episodes | 1 | `phase3_summary.json` vs `phase3_raw.csv` |
| Welch t-tests (HierComm vs MAPPO/CommNet, both regions) | 4 | recomputed from per-seed means |
| Learned-commander ablation, incl. "never significantly improves" | 4 | `wildfire_phase3_hiercomm_learned/` |
| Extended difficulty sweep: best policy per (region, regime) cell | 6 | `wildfire_phase4_extended/robustness_results.csv` |
| Transfer matrix cells re-derived; TRS ratio identity; native diagonal matches main table | 3 | `wildfire_phase6/` |
| Per-file SHA-256 against each frozen `MANIFEST.sha256` | 10 | `wildfire_phase*/MANIFEST.sha256` |

**On the integrity check:** each results directory carries a `FREEZE.json` whose
`fingerprint_sha256` hashes a manifest covering *all* files in that directory — including the
30 model checkpoints. Most checkpoints are excluded from this archive under the 50 MB cap, so
the whole-directory fingerprint **cannot** be recomputed here. Instead every file that *is*
shipped is verified byte-exactly against its recorded hash, and the count of omitted files is
reported. The original fingerprints are printed for the record.

## Tier 2 — run the test suite (~6 seconds, no simulator build)

```bash
pip install -e .        # installs the wildfire_marl package (requirements.txt alone is not enough)
pytest tests/
```

Expected on Linux without the simulator built: **61 passed, 7 skipped**. The 7 skips are the
tests that drive the compiled Cell2Fire binary; they run once you complete Tier 3.

## Tier 3 — re-evaluate a shipped checkpoint against the real simulator (~10–30 min)

Build the simulator first: `third_party/firehose/cell2fire/Cell2FireC/BUILD.md`.

**Only the seed-42 checkpoints are shipped** (the other four seeds' `.pt` files were dropped
under the size cap). The driver is idempotent and *trains* any checkpoint it cannot find, so
you must restrict the run to seed 42 or it will start a multi-hour training job:

```bash
# work on a copy so the frozen artifacts stay pristine
cp -r wildfire_phase3_multiseed /tmp/repro

python scripts/run_phase3.py \
  --methods noop,value_first,greedy_risk,local_reactive,mappo,commnet,hiercomm_heur \
  --regions saudi,california \
  --train-seeds 42 \
  --seeds 42,1042,2042,3042,4042 \
  --episodes 15 \
  --out /tmp/repro \
  --eval-only
```

Compare the regenerated per-episode records against the frozen ones:

```bash
python - <<'PY'
import pandas as pd
frozen = pd.read_csv("wildfire_phase3_multiseed/phase3_raw.csv")
rerun  = pd.read_csv("/tmp/repro/phase3_raw.csv")
f = frozen[(frozen.train_seed == 42) | (frozen.train_seed.isna())]
key = ["region", "policy", "seed", "episode"]
m = f.merge(rerun, on=key, suffixes=("_frozen", "_rerun"))
m["dWEL"] = (m.WEL_frozen - m.WEL_rerun).abs()
print(m.groupby(["region", "policy"]).dWEL.max().sort_values(ascending=False))
PY
```

**Expected:** `dWEL` is exactly `0.0` for every policy except **Greedy-Risk**, which is a
documented, bounded irreproducibility — see the audit note in `docs/RESULTS_FROZEN.md`. Saudi
Greedy-Risk reproduces exactly; California Greedy-Risk lands at WEL 28.32 / ISR 0.1578 against
the frozen 28.40 / 0.1556 (Δ ≤ 0.08 WEL). No ranking, significance test, or conclusion in the
paper is affected. We report this rather than hiding it.

## Tier 4 — full retraining (GPU-hours)

The complete protocol, exactly as run for the paper:

```bash
python scripts/run_phase3.py \
  --methods noop,value_first,greedy_risk,local_reactive,mappo,commnet,hiercomm_heur \
  --regions saudi,california --train-seeds 42,1042,2042,3042,4042 --parallel 6 \
  --train-steps 100000 --episodes 15 --out wildfire_phase3_multiseed
```

Cost: 30 training runs (3 learned methods × 2 regions × 5 seeds) at 100k steps each. On the
6-way-parallel CPU configuration used for the paper this is roughly a day of wall-clock.
Determinism: the reset pipeline is seeded and proven bit-identical on CPU.

---

## Claim → artifact index

| Paper claim | Command | Frozen artifact |
|---|---|---|
| Main results table (WEL/ISR, 7 policies × 2 regions) | Tier 1, or Tier 4 to retrain | `wildfire_phase3_multiseed/phase3_summary.json`, `phase3_main_table.tex` |
| HierComm best-in-class in both regions | Tier 1 | `phase3_raw.csv` |
| Significance: CA MAPPO p=0.0018, CommNet p=0.016; Saudi CommNet p=0.048, MAPPO p=0.32 (n.s.) | Tier 1 | recomputed from `phase3_raw.csv` |
| HierComm has the lowest across-seed variance | Tier 1 | per-seed WEL in `phase3_summary.json` |
| Learned commander never significantly improves on the value-aware rule | Tier 1 | `wildfire_phase3_hiercomm_learned/phase3_summary.json` |
| Ablations (incl. communication is not the driver) | `python scripts/run_phase4_ablations.py --study ablation --ckpt-dir wildfire_phase3_multiseed --out /tmp/abl` | `wildfire_phase4/ablation_summary.json`, `phase4_ablation_table.tex` |
| Difficulty sweep: Local Reactive best on easy, HierComm best on medium/hard | Tier 1 | `wildfire_phase4_extended/robustness_results.csv` |
| Saudi↔California transfer matrix, TRS, CDGG, asymmetry | `python scripts/run_phase6_transfer.py --ckpt-dir wildfire_phase3_multiseed --out /tmp/transfer` | `wildfire_phase6/transfer_summary.json`, `phase6_transfer_table.tex` |
| Generalization sweep | `python scripts/run_phase6_transfer.py --study generalization --ckpt-dir wildfire_phase3_multiseed --out /tmp/gen` | `wildfire_phase6/generalization_summary.json` |
| Dashboard figures equal the paper's numbers | `python scripts/check_dashboard_consistency.py` | `dashboard/dist/data/*.json` |

Note: the Phase-4 and Phase-6 commands above re-evaluate from checkpoints. With only seed 42
shipped they will reproduce the seed-42 slice; the frozen artifacts contain all five seeds.

## Protocol details

- **Training seeds:** 42, 1042, 2042, 3042, 4042. **Eval:** 5 held-out seed groups × 15 episodes.
- **Seed streams are disjoint:** training reset seed = `seed·100000 + episode`; eval reset seed
  = `group + 100000 + episode`. No leakage between train and eval ignitions.
- **Regime** (`default`, defined in `src/wildfire_marl/env/regimes.py`): Saudi wind×0.6 / FFMC 90;
  California wind×0.5 / FFMC 90; `treat_radius=1`, threat ignition, `ros_cv=0.1`,
  `steps_per_action=30`.
- **Aggregation:** per-training-seed mean over episodes first, then mean ± std across the five
  training seeds. Heuristics are deterministic and reported at n=1.

## Known limitations

- Greedy-Risk has a bounded, documented irreproducibility (Tier 3 above).
- QMIX is a failed baseline (degenerate: Saudi ≡ No-Op) and is reported as such, not in the
  main table.
- `FREEZE.json` for Phase 3 records `git_dirty: true`; the manifest fingerprint, not the commit
  hash, is the authoritative reference for the frozen content.
- Full retraining requires the simulator build and is not verifiable from this archive alone
  within a review cycle; Tiers 1–3 are.
