# AAAI-27 Code & Data Supplement — Build Plan

**Target:** one `supplement.zip`, ≤ 50 MB, fully self-contained, fully anonymous.
**Constraint:** no links to our own code/data repos (GitHub, AnonymousGitHub, the live Vercel
dashboard). Links to *third-party public* datasets/tools (Cell2Fire, SRTM, ERA5, FIRMS) are
fine and necessary.

---

## 0. The size problem (measured, not estimated)

The working tree is ~3.5 GB. Where it goes:

| Directory | Size | Verdict |
|---|---:|---|
| `figures/` (phase5/6 GIF renders) | 1,339 MB | **exclude** — regenerable from ckpts |
| `dashboard/` | 1,276 MB | **partially include** (see §3) — 173 MB `node_modules`, 548 MB `public/media`, 552 MB `dist` (of which 548 MB is copied media) |
| `data/` raw rasters (SRTM `.hgt.gz`, DEM `.npy`, NDVI `.tif`) | 703 MB | **exclude** — refetchable, provenance documented |
| `third_party/firehose` | 201 MB | **include source only** (~1.5 MB); 194 MB is a stray `CellsFBP.h.gch` + `.o` build artifacts |
| `wildfire_phase3/4/6*` checkpoints (`.pt`) | 98 MB | **include seed-42 subset only** (~11 MB) |
| `src/ scripts/ configs/ tests/ docs/` | 1.2 MB | **include all** |

Key discovery: **the simulator's actual inputs are tiny.** `data/cell2fire/{California,Saudi}/`
is 0.2 MB total and is everything Cell2Fire needs to run. The 703 MB of rasters are the
*provenance chain* that produced those landscapes, not a runtime dependency.

Second key discovery: `dashboard/dist` minus media is **4.35 MB** — a fully working,
offline, prebuilt dashboard fits in the budget easily.

---

## 1. Directory layout of the supplement

```
supplement/
├── README.md                        ← the reviewer's entry point (see §5)
├── LICENSE                          ← MIT, copyright de-anonymized
├── THIRD_PARTY_LICENSES.md          ← Cell2Fire / Firehose GPL-3.0 attribution
├── requirements.txt                 ← pinned, generated from the live env
├── environment-linux.yml
├── pyproject.toml                   ← authors/URLs stripped
│
├── src/wildfire_marl/               ← 52 .py files, the whole package (0.6 MB)
│   ├── env/                         cell2fire_binding.py, marl_env.py, regimes.py, rewards.py
│   ├── agents/                      heuristics.py, agent_networks.py, strategic_controller.py, ppo_baseline.py
│   ├── train/                       hier_comm_train.py, marl_train.py, hierarchical_train.py
│   ├── eval/                        metrics.py, significance.py, transfer.py, phase3_eval.py, statistics.py
│   ├── infra/                       build_infrastructure.py, cascade.py
│   ├── data/                        to_cell2fire.py, fuel_mapping.py, fwi.py, ignition.py
│   ├── reproducibility/             seeding.py, manifest.py, check_seed_integrity.py, validate_tensors.py
│   └── viz/                         rollout.py, geo_renderer.py, transfer.py, comparison.py
│
├── scripts/                         ← 46 drivers (0.4 MB). Critical ones:
│   ├── run_phase3.py                main results (multi-seed train + eval)
│   ├── run_phase4_ablations.py      ablations
│   ├── run_phase6_transfer.py       Saudi↔California transfer
│   ├── run_phase8_certification.py  final certification gate
│   ├── freeze_results.py            emits FREEZE.json + MANIFEST.sha256
│   ├── reproduce_verification.py    checksum verifier (see §4)
│   ├── build_dashboard_data.py      results → dashboard JSON (fingerprint-gated)
│   ├── check_dashboard_consistency.py   paper ↔ dashboard number check
│   ├── build_report_tables.py / emit_phase8_tables.py   LaTeX tables in the paper
│   ├── fetch_srtm_dem.py            re-downloads the excluded 703 MB
│   └── render_rollout.py / render_phase6_transfer.py    regenerate the excluded figures
│
├── configs/                         ← 14 YAML experiment configs (0.01 MB)
├── tests/                           ← 11 test modules — proves the code runs
│
├── data/
│   ├── cell2fire/California/        ← Forest.asc, Data.csv, Weather.csv, Ignitions.csv,
│   ├── cell2fire/Saudi/                elevation/slope/criticality/asset_type/blast_radius,
│   │                                    fbp_lookup_table.csv, ignition_candidates.json,
│   │                                    conversion_report.json   (0.2 MB — RUNTIME INPUT)
│   ├── processed/{california,saudi}/ ← 7-channel state tensors + per-channel .npy (~0.5 MB)
│   ├── sample/                      ← 2 KB smoke-test tensor
│   └── RAW_DATA_NOT_INCLUDED.md     ← what was dropped, why, and how to refetch
│
├── simulator/                       ← Firehose-patched Cell2Fire, SOURCE ONLY (~1.5 MB)
│   ├── Cell2FireC/*.cpp,*.h,makefile
│   ├── LICENSE (GPL-3.0)
│   └── BUILD.md                     ← `make` instructions + upstream commit SHA
│
├── results/                         ← frozen numbers, NO checkpoints (~3.2 MB)
│   ├── phase3/  phase3_raw.csv, phase3_summary.json, phase3_main_table.tex,
│   │            30× train_curve_*.csv, FREEZE.json, MANIFEST.sha256
│   ├── phase3_hiercomm_learned/  (same shape)
│   ├── phase4/  ablation CSV/JSON/tex + FREEZE.json
│   ├── phase4_extended/  extended robustness JSON/CSV
│   └── phase6/  transfer matrix CSV (1.2 MB), TRS/CDGG JSON, .tex
│
├── checkpoints/                     ← seed-42 only, ~11 MB
│   ├── phase3/  checkpoint_{mappo,commnet,hiercomm_heur}_{saudi,california}_s42.pt   (6 × 1.38 MB)
│   └── phase3_hiercomm_learned/  checkpoint_hiercomm_learned_{saudi,california}_s42.pt (2 × 1.45 MB)
│
├── dashboard/                       ← see §3
│   ├── dist/                        prebuilt, opens offline (4.35 MB incl. data)
│   └── source/                      src/, public/data/, package.json, vite.config.ts, tsconfig*
│
└── docs/
    ├── data_card.md                 dataset provenance
    ├── infra_card.md                critical-asset layer provenance
    ├── RESULTS_FROZEN.md            canonical numbers cited in the paper
    ├── MIGRATION.md                 V1→V2 history (why nothing from V1 is cited)
    └── REPRODUCIBILITY.md           new: claim → command → expected number map (§4)
```

### Deliberately excluded (and stated in the README)

`figures/` GIFs · `data/*/raw` SRTM+NDVI+ERA5 sources · full 5-seed checkpoint sets ·
`node_modules` · `venv`, `venv_wsl` · `.git`, `.github`, `.vercel` · `dashboard/test-results` ·
`reviews/`, `peer_review_report_AAAI2027_HierComm.md`, `peer_phases.md` (internal review
artifacts — also de-anonymizing) · `AAAI Template/` (the paper itself is submitted separately) ·
`notebooks/Data Preliminary/` (contains absolute Windows paths with the author's name — see §2).

---

## 2. Anonymization pass (do this BEFORE zipping — it is disqualifying if missed)

Confirmed leaks found by grep:

| File | Leak | Fix |
|---|---|---|
| `dashboard/.env.local` | **live Vercel OIDC JWT** with `aliakarmas-projects`, project id | **exclude + rotate the token** |
| `.vercel/repo.json` | project/owner ids | exclude the whole `.vercel/` dir |
| `LICENSE:3` | `Copyright (c) 2026 Ali Akarma` | → `Copyright (c) 2026 The Authors` |
| `pyproject.toml:12,79-81` | `authors`, 3× `github.com/aliakarma/wildfire-rl` | strip authors, delete `[project.urls]` |
| `CITATION.cff` | full name + repo URL | **drop the file entirely** |
| `CONTRIBUTING.md:9` | clone URL | drop the file (not needed by reviewers) |
| `data/cell2fire/*/ignition_candidates.json` | `"source": "C:\\Users\\Ali Akarma\\..."` | rewrite to relative path `data/<region>/raw/firms/...` |
| `dashboard/test-results/**` | ~30 files with `C:\Users\Ali Akarma\...` | exclude dir |
| `reviews/history/forensic_audit_report.md` | local paths | exclude dir |
| `notebooks/Data Preliminary/*.ipynb` | paths in outputs | exclude, or strip outputs with `nbstripout` and rewrite paths |
| `README.md` (current) | links to `PEER_REVIEW_REMEDIATION_PLAN.md`, `peer_phases.md` | rewritten wholesale (§5) |

Verification step (run on the staged folder, must return zero hits):

```bash
grep -ril -e "aliakarma" -e "Ali Akarma" -e "vercel.app" -e "VERCEL_OIDC" supplement/
```

Plus: `find supplement -name ".git*" -o -name "*.env*"` must be empty, and unzip the final
archive into a clean dir before checking — not the staging dir.

---

## 3. The dashboard (both halves)

Ship **two** copies so reviewers with no Node install still see it:

**(a) `dashboard/dist/` — prebuilt, zero-install.** 4.35 MB with `data/` included. Reviewer runs
`python -m http.server 8000 --directory dashboard/dist` and opens `localhost:8000`. This is
the headline "click and it works" path. (It must be served over HTTP, not `file://`, because of
the fetch calls — say this explicitly in the README; it's the #1 thing a reviewer will trip on.)

**(b) `dashboard/source/` — auditable.** `src/` (46 files, 0.2 MB), `public/data/` (2.6 MB of
result JSON + replays + basemaps + download CSVs), `package.json`, `package-lock.json`,
`vite.config.ts`, `tsconfig*.json`, `index.html`. No `node_modules` — reviewer runs
`npm ci && npm run dev` if they want to rebuild. Include `e2e/` and `tests/` so the
consistency tests are inspectable.

**Media budget.** `public/media` is 548 MB: 20 GIFs (455 MB), 20 MP4s (30.6 MB), 18 PNG
posters (49 MB), 1 PDF (12 MB).
- Drop all GIFs and the PDF outright — the MP4s are the same content, already committed.
- **Re-encode the 20 MP4s down** with the existing `scripts/compress_media.py` (raise CRF /
  cap width at 720px): 30.6 MB → target ~10 MB, and the media page stays *complete*.
- **Re-encode the 18 posters** PNG → JPEG q80 @ 1200 px: 49 MB → ~1.5 MB.
- Fallback if the re-encode is disappointing: ship a 6-clip subset (both comparison grids +
  4 representative rollouts ≈ 8 MB) and add `dashboard/MEDIA_SUBSET.md` naming the omitted
  clips and the `render_rollout.py` command that regenerates them.

Media lives once, in `dist/media/`, with the source copy symlink-free — don't duplicate it.
Same for the 2.6 MB `public/data`: keep it in `dist/data/` and have the source tree's README
point at it, or accept the 2.6 MB duplicate if that's simpler to script.

---

## 4. Making reproducibility *checkable*, not just claimed

This is what actually moves the reproducibility score. Three additions:

**`docs/REPRODUCIBILITY.md` — a claim→command→number table.** One row per number in the
paper: the table/figure it appears in, the exact command, the runtime, the expected value,
and the file in `results/` that already contains it. E.g.:

| Paper claim | Command | Expect | Frozen artifact |
|---|---|---|---|
| Table 2, HierComm Saudi ΔWEL | `python scripts/run_phase3.py --region saudi --eval-only --ckpt checkpoints/phase3/...s42.pt` | matches `phase3_summary.json` | `results/phase3/phase3_summary.json` |
| Table 4, TRS Saudi→CA | `python scripts/run_phase6_transfer.py --eval-only` | … | `results/phase6/` |
| Ablation: comms n.s. | `python scripts/run_phase4_ablations.py --stats-only` | p > 0.05 | `results/phase4/` |

Be honest in the "cost" column: full multi-seed training is GPU-hours; eval-from-checkpoint is
minutes. Reviewers reward that honesty.

**Three tiers of verification, in escalating cost:**
1. `pytest tests/` — ~1 min, no simulator build needed. Proves the code is real.
2. `python scripts/reproduce_verification.py` — re-derives every table/figure number from the
   frozen CSVs and diffs against `docs/RESULTS_FROZEN.md`. **Seconds.** This is the highest-value
   thing in the whole supplement — a reviewer confirms the paper's numbers without a GPU.
3. `bash simulator/BUILD.md` steps → `python scripts/run_phase3.py --eval-only` on the seed-42
   checkpoints — ~10-30 min, reproduces one column of Table 2 from the actual simulator.

**Integrity:** ship every `FREEZE.json` + `MANIFEST.sha256` already present in the phase dirs,
plus a top-level `SHA256SUMS.txt` over the whole supplement. `scripts/check_dashboard_consistency.py`
lets a reviewer confirm the dashboard's numbers equal the paper's.

**Also state plainly** what a reviewer *cannot* rerun from the zip (full 5-seed training,
raw-raster → landscape conversion) and exactly which script does it if they fetch the data.

---

## 5. README.md rewrite — the reviewer's 60-second path

The current README is a *project* README: research bet, phase plan, links to
`PEER_REVIEW_REMEDIATION_PLAN.md` and `peer_phases.md` (both deleted), a GitHub-flavored
"getting started". Wrong audience and it de-anonymizes. Replace entirely with a
*supplement* README, structured so a reviewer gets value at every depth they choose to go:

**Section order (deliberate — cheapest verification first):**

1. **Title + one-paragraph scope.** What the paper claims, what this archive contains, what it
   deliberately omits and why (size cap). No author names, no repo links, no live URLs.
2. **"Verify in 60 seconds — no install."**
   `python -m http.server 8000 --directory dashboard/dist` → every headline number, chart,
   and replay in the browser, offline. Screenshot-free, one command. Flag the `file://` caveat.
3. **"Verify the numbers in 2 minutes."** `pip install -e .` → `python scripts/reproduce_verification.py`
   → prints the paper's tables recomputed from frozen CSVs with a pass/fail per claim.
4. **"Run the tests."** `pytest tests/` — expected pass count.
5. **Full reproduction path.** Build Cell2Fire (`simulator/BUILD.md`, Linux/WSL only — say this
   loudly, native Windows is unsupported), then eval-from-checkpoint, then full training with
   honest GPU-hour costs and exact seeds (42/1042/2042/3042/4042).
6. **Repository map.** A ~20-line annotated tree — every top-level dir, one line each, so a
   reviewer looking for "where is the reward function" finds `src/wildfire_marl/env/rewards.py`
   without grepping.
7. **Claim → artifact index.** Table mapping each paper table/figure number to the file in
   `results/` and the script that produced it. This is what a reproducibility-focused reviewer
   opens first — put it above the fold in spirit, link to it from §1.
8. **What is NOT included, and why.** Raw 703 MB rasters (refetch: `scripts/fetch_srtm_dem.py`,
   ERA5/FIRMS instructions in `docs/data_card.md`); GIF renders (regenerate: `render_rollout.py`);
   4 of 5 seeds' checkpoints (only s42 shipped; all 5 seeds' *results* are in the CSVs).
   Framing matters: "omitted under the 50 MB cap, regenerable via X" reads very differently
   from silence.
9. **Environment.** Exact Python version, pinned `requirements.txt`, OS (Linux/WSL2), GPU used,
   approximate wall-clock per phase.
10. **Licensing.** MIT for our code; Cell2Fire/Firehose GPL-3.0 vendored in `simulator/` with
    its own LICENSE; data source licenses (SRTM public domain, ERA5 Copernicus, FIRMS NASA).

**Tone rules:** imperative commands in copy-pasteable fenced blocks, every path relative to the
zip root, no "see our repo", no first-person author references, and an explicit statement that
the archive is self-contained.

Keep a **second, short `dashboard/README.md`** covering just the two run modes and what each
of the 8 pages shows.

---

## 6. Build script

Add `scripts/build_supplement.py` so the archive is reproducible and not hand-assembled:

1. Stage to a temp dir via an explicit **allowlist** (never a denylist — a denylist is how
   `.env.local` ends up in the zip).
2. Copy + rewrite: apply the §2 anonymization edits on the staged copies only, never in-place
   on the working tree.
3. Re-encode media (§3), regenerate `dashboard/dist` from the trimmed media.
4. Emit `SHA256SUMS.txt`, run the grep-based leak check, hard-fail on any hit.
5. Zip with `-9`, print the final size, hard-fail above 48 MB (2 MB safety margin).
6. Unzip to a scratch dir and run `pytest tests/ -q` + `reproduce_verification.py` inside it —
   proves the *archive*, not the repo, is what works.

## 7. Projected size

| Component | Uncompressed |
|---|---:|
| `src/ scripts/ configs/ tests/` + root files | 1.1 MB |
| `data/` (cell2fire landscapes + processed tensors + sample) | 1.0 MB |
| `simulator/` (Cell2Fire source, no build artifacts) | 1.5 MB |
| `docs/` | 0.1 MB |
| `results/` (CSV/JSON/tex/logs, no `.pt`) | 3.2 MB |
| `checkpoints/` (seed 42 only, 8 files) | 11.0 MB |
| `dashboard/dist` (no media) + `dashboard/source` | 6.0 MB |
| media (20 MP4s re-encoded + 18 JPEG posters) | 11.5 MB |
| **Total staged** | **~35 MB** |
| **Zipped** (`.pt`/`.mp4` don't compress; text does) | **~28–30 MB** |

Comfortable headroom. If the re-encode underperforms, the media subset (§3 fallback) buys
back another 3-4 MB, and dropping `phase6`'s 1.2 MB transfer CSV to a gzip buys a little more.

## 8. Order of operations

1. Rotate the leaked Vercel token (do this first, independent of the deadline).
2. Write `scripts/build_supplement.py` with the allowlist.
3. Write the new `README.md` and `docs/REPRODUCIBILITY.md`.
4. Vendor `simulator/` from `third_party/firehose/cell2fire`, source files only, + `BUILD.md`.
5. Re-encode media, rebuild `dashboard/dist`.
6. Run the build script; check size + leak grep.
7. Unzip clean, follow your own README top to bottom as if you were the reviewer. Fix whatever
   was ambiguous. This step catches more than the other six combined.
