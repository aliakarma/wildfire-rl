# Frozen result artifacts

Every number in the paper and on the dashboard is derived from one of the directories below.
Each is **frozen**: `MANIFEST.sha256` lists a SHA-256 per file and `FREEZE.json` records a
single fingerprint over that manifest, so any edit is detectable.

| Directory | Contents | Fingerprint |
|---|---|---|
| `wildfire_phase3_multiseed/` | Main results — 7 policies × 2 regions × 5 training seeds; per-episode CSV, summary JSON, LaTeX table, training curves, logs, 30 checkpoints | `da4a381d…` |
| `wildfire_phase3_hiercomm_learned/` | Learned-commander ablation at the same 5-seed protocol | `92562477…` |
| `wildfire_phase4/` | Single-factor ablations (communication, hierarchy, reward shaping, RL fine-tuning) | `be6d0e21…` |
| `wildfire_phase4_extended/` | Extended easy/medium/hard difficulty sweep across all seven policies | `cada146a…` |
| `wildfire_phase6/` | Saudi↔California transfer matrix and held-out generalization sweep | `cfce7608…` |

Model checkpoints (`*.pt`) live in these directories on disk but are **not tracked by git** —
they ship with the AAAI supplement instead. Everything else here is committed.

## Verifying

```bash
# Re-derive every headline claim from the frozen CSVs (seconds, no GPU, no simulator build).
python scripts/verify_supplement.py

# Recompute one directory's fingerprint; it must match the table above.
python scripts/freeze_results.py results/wildfire_phase3_multiseed
```

Fingerprints are computed over paths **relative to each directory**, so the artifacts verify
identically regardless of where the tree is checked out.

## Provenance

`docs/RESULTS_FROZEN.md` is the authoritative record: protocol, seed derivation, reproduction
commands, significance tests, and the documented reproducibility caveats.
