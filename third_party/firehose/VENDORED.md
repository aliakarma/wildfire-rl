# VENDORED COPY — provenance & pruning record

- **Upstream:** https://github.com/aidan-curtis/firehose
- **Commit:** `e49a52a9a987a1585d25ddca2f711222ca500ffb` (2022-05-12, upstream HEAD at vendoring time)
- **Vendored:** 2026-07-04 (Phase 1 of `REMEDIATION_PLAN_V2.md`), `.git` removed → plain files
- **License:** GPL-3.0 (see `LICENSE`)
- **Paper:** "Firehose: Wildfire Prevention and Management using Deep Reinforcement Learning"
  (Shen, Curtis, et al.) — https://williamshen-nz.github.io/firehose/

## Why vendored (not a submodule)

The plan calls for "a reference fork you own": Firehose is unmaintained (2022, old `gym` 0.21)
and we need its tree present verbatim for (a) building the *interactive* Cell2Fire binary
(`cell2fire/Cell2FireC/`, stdin/stdout step protocol) and (b) reading its wrapper code as the
reference for our own Gymnasium binding (`src/wildfire_marl/env/cell2fire_binding.py`). We do
NOT install or run its Python package (old-gym rot); it is reference + C++ source only.

## Pruned from the upstream tree (size: 193 MB → ~7 MB)

| Removed | Size | Why |
|---------|------|-----|
| `.git/` | 57 MB | vendored as plain files; provenance recorded here instead |
| `data/*` except `Sub20x20/`, `Sub40x40/`, `Harvest40x40/` | ~156 MB | only the stock benchmark maps used in Phases 1/4 are needed |
| `pretrained_models/` | 17 MB | their SB3 checkpoints — never cited/used (and `.zip` is repo-ignored) |
| `figs/` | 3.8 MB | paper figures/GIFs, not needed |
| `scratch/` | 11 MB | upstream scratch space, not needed |

Everything else (in particular `cell2fire/` — the Python wrapper reference and the patched
`Cell2FireC` C++ sources — plus `README.md`, `LICENSE`, `requirements.txt`, experiment shell
scripts, analysis notebooks) is byte-identical to the upstream commit above.

## Local modifications

**None.** This tree is frozen reference code. The physics-integrity diff against upstream
Cell2Fire is documented in `../README.md`. Build artifacts (`*.o`, the `Cell2Fire` binary) are
produced in-place by `make` and are git-ignored.
