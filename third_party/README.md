# third_party/ — external simulator code (Phase 1, REMEDIATION_PLAN_V2)

| Tree | What | How it is tracked |
|------|------|-------------------|
| `Cell2Fire/` | **Upstream Cell2Fire** (Pais et al.) — the peer-reviewed, physics-validated cell-based fire growth simulator. Used to build/run the *stock* simulator and as the pristine physics reference. | git **submodule** → https://github.com/cell2fire/Cell2Fire |
| `firehose/` | **Firehose** (Shen, Curtis et al., 2022) — the RL benchmark built on a Cell2Fire fork whose C++ main loop was patched for *per-step interactive control* (pause each period, read harvest actions on stdin, emit per-period grid CSVs). Our Gymnasium binding is modeled on its protocol and uses its interactive binary. | **vendored copy** (pruned) — see `firehose/VENDORED.md` |

## Physics-integrity evidence (fork vs upstream)

Verified 2026-07-04 by diffing `firehose/cell2fire/Cell2FireC/` against upstream at the fork-base
commit `336119d` (the last upstream change to `CellsFBP.cpp` before the May-2022 fork):

- **Byte-identical:** `FBPfunc5_NoDebug.c`, `FBP5.0.h` (the Canadian FBP fire-behavior equations),
  `SpottingFBP.cpp`, `Ellipse.cpp`, `Forest.cpp`, `Lightning.cpp`, `ReadCSV.cpp`, `WriteCSV.cpp`.
- **`CellsFBP.cpp`:** 45 substantive changed lines, *all* of the form
  `if (args->verbose)` → `if (should_print && args->verbose)` — debug-logging suppression only.
  Zero fire-behavior changes.
- **`Cell2Fire.cpp` / `ReadArgs.*`:** the interactive-control patch (`--steps-action`,
  `--steps-before`, `--HarvestPlan`, stdin action protocol) — simulation *loop control and I/O*,
  not fire spread.

Conclusion: **the interactive binary's fire physics is unmodified Cell2Fire.** (Note: upstream
`HEAD` later added an optional configurable spread radius, 2026-03; that post-fork feature is not
part of the fork and defaults to the classic 8-neighbour behavior upstream as well.)

## Rules

- Never edit either tree (except building object files in place, which stay untracked).
  Build-time configuration goes through `make` variable overrides (e.g. `EIGENDIR=...`), not
  source edits. Cell2Fire physics is never modified — all suppression/agent logic lives in
  `src/wildfire_marl/env/`.
- Licensing: both trees are **GPL-3.0**. They are build/runtime dependencies of the experiments;
  release implications are handled in Phase 15 (documented, fine for academic release).
