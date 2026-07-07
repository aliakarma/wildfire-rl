# peer_phases.md — Execution Log for PEER_REVIEW_REMEDIATION_PLAN.md

This file records the phase-by-phase execution of `PEER_REVIEW_REMEDIATION_PLAN.md` (the
remediation of the AAAI 2027 peer-review findings in `AAAI Template/Peer_Review_Report.md`).
One section is appended per completed phase: actions taken, evidence, root causes, deviations
from the plan, and the success-gate verdict.

**Repository:** `wildfire-rl` · **Branch:** `additional` · **Executor:** automated (supervised)

---

# Phase 0 — Forensic Audit & Ground Truth

**Status:** ✅ COMPLETE · **Date:** 2026-07-07 · **Effort:** 1 session (audit only, no manuscript edits)

## Objective (from the plan)

Establish the provenance of every number in the manuscript before anything is rerun or
rewritten. Root-cause the four artifact anomalies (C1 Table 1/Table 3 contradiction, H1
MAPPO ≙ No-Op, H8 identical ablation rows, H3 QMIX ghost results), verify the simulator
facts (M4 fire cadence, M5 Cell2Fire boundary) from code, and record the reframing strategy.

## Deviation from the plan

The plan called for the provenance table to live in `docs/paper_provenance.md`. Per user
instruction, all Phase 0 output (including the provenance table) is recorded here instead.
No other deviations.

## Method

Every table, figure, and in-text statistic in `AAAI Template/AnonymousSubmission2027.tex`
was traced against the artifacts under `results/` (phase8, phase8_smoke, phase13, phase15,
runs), the checkpoint files' **embedded training configs** (loaded with torch in the WSL
venv), the training/eval scripts (`scripts/`, `src/wildfire_marl/`), and the environment
source code. Figures were compared by SHA-256. Verdicts: `TRACED` (matches an artifact),
`PROTOCOL-MISMATCH` (real artifact, incomparable protocol), `CONTRADICTED` (artifact says
otherwise), `UNTRACEABLE` (matches nothing in the repository).

---

## 1. Provenance Table

### Table 1 (Main results) — ✅ TRACED

Source: `results/phase8/multiseed_eval_aggregate_{saudi,california}.csv` +
`results/phase8/certification_summary_{saudi,california}.json`.

| Paper value | Artifact value | Verdict |
|---|---|---|
| Saudi Hierarchical WEL 20.28 [19.8, 20.9] | mean of seed means (20.6, 19.8, 21.4, 19.8, 19.8) = 20.28; CI [19.8, 20.92] | TRACED |
| Saudi Value-First WEL 19.48 [19.0, 20.1] | (19.0, 19.0, 20.6, 19.0, 19.8) = 19.48; CI [19.0, 20.12] | TRACED |
| Saudi Flat MARL / No-Op WEL 27.00, ISR 0.286 | 27.0 at every seed, std = 0 | TRACED |
| California Hierarchical WEL 24.53 [22.9, 26.1] | (24.2, 26.53, 26.53, 22.0, 23.4) = 24.533; CI [22.88, 26.07] | TRACED |
| California Flat MARL / No-Op WEL 25.25 | (24.6, 27.73, 27.73, 22.4, 23.8) = 25.253 | TRACED |
| All ISR, Reward, CE cells | match certification JSONs to reported precision | TRACED |

Protocol confirmed from raw data: `multiseed_eval_raw_*.csv` = 375 rows per region =
5 policies × 5 seeds (42, 1042, 2042, 3042, 4042) × 15 episodes. **Statistical unit = 5 seed
means (n = 5)** — this resolves M8 (the reviewer's question about n) definitively.

### Table 2 (Compliance diagnostics) — ⚠️ PARTIALLY TRACED

Computed by `scripts/run_phase8_certification.py` (calls
`wildfire_marl.eval.compliance.analyze_compliance_trajectory`), which **prints** Target
Compliance Rate / Sector Drift / Dispatch Latency to stdout and saves only the supporting
plots (`trajectory_overlay_*.png`, `sector_occupancy_*.png`, `target_distance_*.png` in
`results/phase8/`). The specific numbers 1.000 / 4.32 / 1.67 / 4.18 / 1.72 are **not archived
in any file**. Recomputable, but currently unverifiable. → Re-run and archive in Phase 4.

### Table 3 (Ablations) — ⚠️ TRACED but PROTOCOL-MISMATCH (root cause of C1, see §2.1)

Source: `results/phase8/ablation_results_saudi.csv` — every paper value (17.4/0.400,
27.0/0.286, 18.2/0.390 ×3, 17.4/0.400) appears verbatim. But the artifact has **no seed
column**: `scripts/run_ablation_v2.py` defaults are `--seed 42` (single seed),
`--episodes 15`, and it **retrains each configuration from scratch** (40 BC episodes +
120 RL episodes) before evaluating. A California ablation CSV also exists
(`ablation_results_california.csv`) but is not reported in the paper.

### Table 4 (Hyperparameters) — ❌ CONTRADICTED on 6 of 8 checkable hierarchical entries

The shipped checkpoints embed their training configs. `checkpoint_hierarchical_saudi.pt`
(SHA-256 `43a437aff0ee402e…` — **the very hash the paper publishes in Appendix D**) says:

| Parameter | Paper (Table 4) | Checkpoint config | Verdict |
|---|---|---|---|
| BC episodes | 500 | `pretrain_episodes: 40` | CONTRADICTED |
| BC epochs | 50 | `bc_epochs: 100` | CONTRADICTED |
| RL episodes | 200 | `hierarchical_episodes: 120` | CONTRADICTED |
| KL coefficient λ | 0.1 | `kl_coef: 0.3` | CONTRADICTED |
| Entropy coefficient β | 0.01 | `ent_coef: 0.05` | CONTRADICTED |
| Commander LR | 3 × 10⁻⁴ | `rl_lr: 5e-05` | CONTRADICTED |
| ISR weight α | 20 | `isr_weight: 20.0` | TRACED |
| MAPPO training steps | 10,000 | `total_steps: 10000` | TRACED |

Note: `configs/marl/mappo_*.yaml` says `total_steps: 50000`, but both MAPPO checkpoints
embed `total_steps: 10000` — the value was overridden at training time. The paper's 10,000
is accurate to the artifact; the config file is misleading. The §4.3 text (λ = 0.1,
β = 0.01, K = 500) repeats the wrong Table 4 values.

### Table 5 (Appendix B, per-seed WEL) — ❌ UNTRACEABLE (fabricated; NEW issue C6)

| Region | Paper per-seed values | Actual per-seed values (aggregate CSV) |
|---|---|---|
| Saudi | 19.8, 20.4, 20.9, 20.2, 20.1 (mean 20.28) | 20.6, 19.8, 21.4, 19.8, 19.8 (mean 20.28) |
| California | 24.1, 25.0, 24.8, 23.9, 24.9 (mean 24.54) | 24.2, 26.53, 26.53, 22.0, 23.4 (mean 24.53) |

The paper's values match the artifact **means** but not the artifact **values**, and they
materially understate the variance (Saudi actual std 0.72 vs paper-implied 0.41; California
actual std 1.99 vs paper-implied 0.49). Reviewer #1 cross-checked Table 5 against Table 1
and praised the consistency — that check passed only because the invented values were
constructed to preserve the mean. This is a serious integrity defect the review did not
catch, and it is worse than any issue the review did raise.

### Table 6 (Appendix C, statistical tests) — ⚠️ MIXED

Source: `certification_summary_*.json` → `comparisons.WEL`.

| Row | Paper | Artifact | Verdict |
|---|---|---|---|
| Saudi vs Value-First | t = −1.94, p = 0.115, d = −1.23 | t = 1.768, p = 0.11508, d = 1.118 | p TRACED; **t and d UNTRACEABLE** (magnitudes match nothing in the repo; plausibly adjusted to fit the fabricated Table 5) |
| Saudi vs Flat MARL | t = 21.0, p = 3.0 × 10⁻⁵, d = 13.28 | t = −21.0, p = 3.039 × 10⁻⁵, d = −13.28 | TRACED (sign flipped for presentation) |
| California vs Value-First | t = −0.062, p = 0.952, d = −0.039 | t = −0.0619, p = 0.9521, d = −0.0392 | TRACED (exact, including sign) |
| California vs Flat MARL | t = −0.517, p = 0.620, d = −0.327 | t = −0.5170, p = 0.6196, d = −0.3270 | TRACED (exact, including sign) |

The sign convention is also inconsistent across rows (Saudi flat-MARL row flipped, both
California rows not flipped).

### Figures 2, 3, 4 — ❌ ALL THREE ARE RENAMED SMOKE-TEST ARTIFACTS (NEW issue C7)

SHA-256 comparison:

| Paper figure | Byte-identical to | Implication |
|---|---|---|
| `Figures/comparison_flat_vs_hier.png` (Fig. 2) | `results/phase13/smoke_comparison_noop_vs_hier.png` | The figure **is** a No-Op-vs-Hierarchical smoke-test comparison, renamed to claim Flat MARL. The caption's "Flat MARL" is false — the panel title "No-Op Rollout (Saudi)" that Reviewer #1 spotted is the original truth. Root cause of H4. |
| `Figures/commander_decisions.png` (Fig. 3) | `results/phase13/smoke_commander_decisions.png` | Smoke-test dispatch snapshot, not from a certified run — explains the stars-at-agent-positions anomaly (M2). |
| `Figures/transfer_heatmap.png` (Fig. 4) | `results/phase13/smoke_transfer_heatmap.png` | Smoke-test transfer heatmap. The certified transfer artifact (`results/runs/transfer_heatmap.png`) is a **different** file the paper does not use. |

None of the three figures in the submission derive from the certified 5-seed protocol.

### §9 Transfer results — ❌ BROKEN EXPERIMENT (extends M1)

`results/runs/transfer_results_summary.csv` (the non-smoke artifact):

- TRS values are 1.0 and 0.9999999997 — not the 0.95/0.96 shown in Figure 4 (which is the
  smoke artifact). The paper's TRS numbers trace to neither.
- Cross-region rows are **identical to 10 decimal places** with native rows
  (Saudi→California reward −379.9583… = California→California reward −379.9583…), and the
  "native Saudi" row reports WEL 27.0 = No-Op level (vs. Table 1's 20.28). The transfer
  evaluation evaluated policies that had no measurable effect, or mis-loaded them. Either
  way §9 is built on a broken run, independent of the metric-insensitivity critique.

### Appendix D (checkpoint hashes) — ✅ TRACED

All six SHA-256 prefixes in the paper match `results/phase15/reproducibility_certificate.json`
and the phase8 manifests, and the six checkpoint files exist in `results/runs/`. The QMIX
checkpoints are real (see §2.4).

### In-text statistics

- Abstract/§6 p-values and d — TRACED to certification JSONs (with the C2 caveat that the
  California flat-MARL non-result, p = 0.620, is in the artifact and in Appendix C but
  contradicts the abstract's blanket claim).
- §7 compliance numbers — same status as Table 2 (stdout-only).
- §9 "TRS > 0.95" — UNTRACEABLE (smoke figure shows ≥ 0.95; certified artifact shows ≈ 1.0).

---

## 2. Root Causes

### 2.1 C1 — Table 1 vs. Table 3 contradiction: ROOT-CAUSED

They are different experiments presented as the same system:

- **Table 1** (WEL 20.28): the shipped checkpoint `checkpoint_hierarchical_saudi.pt`,
  evaluated under the certified protocol — 5 seeds × 15 episodes
  (`scripts/run_multiseed_eval.py` → `results/phase8/`).
- **Table 3** (WEL 17.4): a **freshly retrained** commander produced inside
  `scripts/run_ablation_v2.py` (40 BC + 120 RL episodes, seed 42 only), evaluated for
  15 episodes at that single seed.

Different checkpoint, different seed protocol → the numbers were never comparable. Nothing
in the paper discloses this. Fix (plan Phase 3): evaluate the ablation table with the same
checkpoints-and-5-seed protocol as Table 1, making the Full System row identical by
construction.

### 2.2 H1 — Flat MAPPO ≙ No-Op: ROOT-CAUSED (two compounding causes)

1. **Budget.** The checkpoint's embedded config confirms `total_steps: 10000` — ≈ 66
   episodes of 150 steps. The config file's 50,000 was overridden at training time.
2. **Objective mismatch.** `train_mappo()` trains on the environment's shared micro reward =
   `FireSizeReward` (Firehose's −burning-cells penalty) minus coordination and compliance
   penalties (`marl_env.py:262-275`). It was **never trained on the WEL/ISR objective**
   (Eq. 1) it is evaluated on. Reviewer #2's objective-level confound is confirmed as fact.

Behavioral evidence from the certified eval: MAPPO agents **do act** (CE = 0.975 Saudi /
0.894 California proves treatment actions occur), but across all 75 episodes per region
their treatments changed **not a single asset outcome** (WEL/ISR per-seed identical to
No-Op in both regions; burned_cells identical in California, within ~1 cell in Saudi).
Verdict: *genuinely trained, genuinely ineffective policy* — not an evaluation-pipeline
bug (the eval demonstrably loads and runs the actor; its CE differs from No-Op's 1.0).
"Coordination collapse" as a general claim remains confounded by (1) and (2); Phase 2 must
retrain with a fair budget on the matched objective and log action histograms.

### 2.3 H8 — Identical ablation rows: ROOT-CAUSED

The byte-identical values (18.2/0.390 for three unrelated ablations; entropy ablation
identical to full system) are genuinely present in `ablation_results_saudi.csv` — not a
transcription error. Mechanism: **single-seed protocol + quantized outcome space**. WEL on
Saudi takes discrete values on a ≈0.8-granularity lattice (17.4, 18.2, 19.0, 19.8, …)
because it is a sum over a small set of discrete asset values; with one seed and 15
episodes, distinct configs readily collide on the same lattice point. The 5-seed rerun
(plan Phase 3) dissolves this.

### 2.4 H3 — QMIX ghost results: ROOT-CAUSED (worse than the review assumed)

QMIX was **trained** (checkpoints embed `algo: qmix`, `total_steps: 10000`) **and
evaluated** — `results/runs/marl_eval_summary_{saudi,california}.csv` contains QMIX rows:
Saudi WEL 27.0 (= No-Op), CE 0.56; California WEL 24.35, CE 0.37. The results exist and
were omitted from the paper while the checkpoint hashes were kept in Appendix D. The same
files show an unreported "Heuristic" policy achieving **California WEL 15.15** — far better
than the paper's best California number (24.53) — under that (older, 20-episode) protocol.
Phase 2 must either substantiate or retire that result; if a simple local heuristic really
beats everything in California, the paper's story changes materially.

### 2.5 M4 — Fire-update cadence: RESOLVED IN THE PAPER'S FAVOR (labeling error)

Ground truth from `single_agent_env.py` (docstring + `Cell2FireBinding` wiring): one fire
period = **1 simulated minute**; `steps_per_action = 60` means each agent step advances the
simulator by 60 fire periods (1 simulated hour; weather also advances hourly). The fire
therefore evolves **every single agent step** — 150 fire advances per episode, not ≤ 2.
Reviewer #1's inference (W7) is factually wrong, but it was caused by Table 4's mislabeled
row "Steps per fire update = 60". Fix (Phases 6/8): relabel to "Simulated minutes (fire
periods) per agent step: 60" and state "1 step = 1 simulated hour" in §5. The "dynamically
evolving hazard landscape" framing is defensible as-is.

### 2.6 M5 — Cell2Fire vs. wrapper boundary: DOCUMENTED FROM CODE

- **Cell2Fire proper** (C++ subprocess via `env/cell2fire_binding.py`, `third_party/`):
  FBP fuel models, wind/slope-driven spread, fire periods, `ros_cv` stochasticity.
- **Custom wrapper** (`env/single_agent_env.py`, `env/marl_env.py`, `infra/cascade.py`):
  treatment patches, the **asset-detonation cascade** (`cascade_prob = 0.1` per-neighbor
  ignition when an asset cell burns — this is what Table 4 calls "cascade probability";
  it is not an FBP concept, exactly as the reviewers suspected), infrastructure reward
  shaping, action masking, PettingZoo multi-agent layer.
- Appendix E must attribute cascade/treatment/rewards to the wrapper, not to Cell2Fire.

### 2.7 H5 — GIS provenance: EXISTS IN-REPO, ABSENT FROM PAPER

`docs/data_card.md` already documents: MODIS NDVI → FBP fuel-class mapping (with
literature citations and thresholds), ERA5 June-2025 hourly weather + CFFDRS FWI codes,
the deterministic converter (`wildfire_marl.data.to_cell2fire`), and a hashed data manifest
(`results/data_manifest.json`). Phase 6 is largely a **transfer-into-paper** task. One
declared data debt: the raw California NDVI GeoTIFF was not archived (an assumed NDVI
range is documented in the data card) — must be disclosed or re-downloaded. Bonus finding:
**64×64 grids already exist for both regions** in `data/*/grids/64x64/`, so the Phase 5
scale study is data-ready.

---

## 3. New Issues Registered (not in the peer review)

These extend the plan's issue register and are assigned to phases:

| ID | Issue | Evidence | Assigned phase |
|----|-------|----------|----------------|
| **C6 (NEW, Critical)** | Appendix B per-seed table fabricated (mean-preserving invented values; variance understated ~2–4×); Appendix C Saudi-vs-Value-First t/d (−1.94/−1.23) untraceable | §1 Table 5 / Table 6 rows | Phase 3 (regenerate), Phase 4 (recompute), Phase 8 (rewrite) |
| **C7 (NEW, Critical)** | All three paper figures are renamed smoke-test artifacts; Fig. 2's rename (`noop`→`flat`) misrepresents its content | SHA-256 identity, §1 | Phase 7 (regenerate all from certified runs) |
| **C8 (NEW, Critical)** | Table 4 / §4.3 hyperparameters contradict the shipped checkpoints' embedded configs on 6 entries | §1 Table 4 | Phase 6/8 (report checkpoint-config values) |
| **H9 (NEW, High)** | Transfer experiment itself broken (cross-region ≡ native to 10 d.p.; native Saudi at No-Op level), independent of TRS insensitivity | §1 transfer row | Phase 5 (redo §9 from scratch) |
| **M11 (NEW, Medium)** | Unreported evaluated baselines: QMIX rows and a "Heuristic" with California WEL 15.15 (beats every paper policy) in `marl_eval_summary_*.csv`; "Greedy-Risk Heuristic" evaluated in phase8 aggregate but dropped from the paper | §2.4 | Phase 2 (evaluate under main protocol; report or retire with rationale) |
| **M12 (NEW, Medium)** | Table 2 compliance numbers exist only in stdout (never archived) | §1 Table 2 | Phase 4 (recompute + archive CSV) |

**Integrity rule adopted for all subsequent phases** (consequence of C6/C7/C8): every
number, table, and figure in the revised manuscript must be generated programmatically from
certified artifacts (extend `scripts/build_report_tables.py`); nothing is hand-entered.

---

## 4. Strategy Decision (recorded per plan §2)

**Adopted: the Area Chair's reframe-plus-rerun path.** The paper is repositioned as an
infrastructure-aware wildfire-coordination **benchmark** (documented GIS pipeline, Cell2Fire
backend, hashed data manifest — assets that genuinely exist in this repo) plus a
**diagnostic study** of flat-CTDE failure under fair baselines, with the BC+KL hierarchy as
a **reference solution**. Phase 0 evidence reinforces this choice: the learned hierarchy's
edge over its teacher is nonexistent (known), the flat-MARL diagnosis is currently
confounded (§2.2), and the repo's strongest verifiable assets are the environment/data
pipeline and the reproducibility tooling. The Phase 2 gate will decide between
"diagnosis + benchmark" (if fair baselines still fail) and "benchmark + reference
solutions" (if they don't).

---

## 5. Rerun Schedule for UNTRACEABLE / CONTRADICTED items

| Item | Status | Scheduled fix |
|---|---|---|
| Table 5 per-seed values | UNTRACEABLE (fabricated) | Phase 3: export real per-seed table from rerun artifacts |
| Table 6 Saudi-vs-VF t, d | UNTRACEABLE | Phase 4: recompute with declared unit (n=5 seed means), uniform sign convention |
| Table 4 hierarchical hyperparameters (6 entries) | CONTRADICTED | Phase 6/8: report checkpoint-embedded configs verbatim |
| Figures 2, 3, 4 | UNTRACEABLE to certified runs (smoke renames) | Phase 7: regenerate from certified rollouts |
| §9 TRS values + Figure 4 | UNTRACEABLE + broken source run | Phase 5: redesign and rerun transfer/generalization |
| Table 2 compliance stats | Not archived | Phase 4: recompute + archive |
| Table 3 (all rows) | PROTOCOL-MISMATCH | Phase 3: 5-seed × 15-episode rerun incl. BC-only row |
| Flat MARL rows (Tables 1, 3, 6) | Confounded baseline | Phase 2: retrain (fair budget, matched WEL/ISR objective, curves) |
| QMIX | Trained+evaluated, unreported | Phase 2: evaluate under main protocol; report or delete hashes |

---

## 6. Success-Gate Verdict

| Gate criterion (from plan Phase 0) | Result |
|---|---|
| Provenance table covers 100% of reported numbers, no blanks | ✅ §1 — every table, figure, and in-text statistic classified |
| C1 divergence has a written, artifact-reproducible root cause | ✅ §2.1 — different checkpoint + single-seed retraining protocol in `run_ablation_v2.py` |
| H1 bit-identity root cause distinguishes collapse from eval bug | ✅ §2.2 — trained-but-ineffective policy (CE < 1 proves the actor runs); 10k steps + objective mismatch confirmed from checkpoint config and trainer source |
| H8 identical-values anomaly root-caused | ✅ §2.3 — single seed + quantized WEL lattice |
| Every UNTRACEABLE number scheduled for rerun | ✅ §5 |
| Fire cadence + Cell2Fire boundary documented from code | ✅ §2.5, §2.6 — cadence resolves in the paper's favor (labeling fix); cascade is wrapper, not FBP |
| Strategy decision recorded | ✅ §4 |

**GATE: PASS.** Phase 1 (desk-reject elimination) and Phase 2 (baseline rehabilitation)
are cleared to start; per the plan they may run in parallel (Phase 1 is manuscript
hygiene on Windows, Phase 2 is compute in WSL).

### Headline takeaways for the authors

1. The certified 5-seed evaluation pipeline (Table 1) is real and fully traceable — the
   paper's strongest evidentiary asset.
2. Three classes of content were **not** generated from certified artifacts: the per-seed
   appendix (invented), all three figures (smoke-test renames), and six hyperparameter
   entries (contradict the shipped checkpoints). These are integrity defects that must
   never reach an external reviewer; the programmatic-generation rule in §3 exists to make
   them structurally impossible.
3. The two claims the review attacked hardest (flat-MARL collapse, transfer) sit on a
   10k-step objective-mismatched baseline and a broken transfer run respectively — both
   require the new experiments the plan already schedules, not wording changes.
4. Two review objections dissolve on inspection: the fire is dynamic (M4 was a label
   error), and the GIS provenance exists in `docs/data_card.md` (H5 is a transfer task).

---

# Phase 1 — Desk-Reject Elimination (LaTeX, Bibliography, Checklist)

**Status:** ✅ COMPLETE · **Date:** 2026-07-07 · **Ran in parallel with Phase 2 training**

## Objective (from the plan)

Remove every mechanical ground for desk rejection: LaTeX preamble violations (C5), the
undefined citation (C4), equation overflow (L2), TRS wording (L3), figure-path verification
(L4), and the blank reproducibility checklist (C3, skeleton stage).

## Actions Taken

### 1. Preamble compliance (C5) — `AnonymousSubmission2027.tex`

- `\usepackage{aaai2027}` → `\usepackage[submission]{aaai2027}` (anonymized review format;
  also restores `\pdfinfo` handling).
- **Removed** `\usepackage{times}`, `\usepackage{helvet}`, `\usepackage{courier}` — the
  style loads `newtxtext`/`helvet`/`courier` itself and its comments explicitly forbid
  loading `times` (verified in `aaai2027.sty` lines 64–73).
- **Removed** manual `\setlength{\pdfpagewidth/\pdfpageheight}` — the style owns geometry.
- **Removed** `\usepackage{caption}` and `\usepackage{subcaption}` — both unused in the
  document (no `\captionsetup`, no `subfigure` environments) and caption-altering packages
  are disallowed by the AAAI kit.
- Remaining package audit: `url`, `graphicx`, `natbib`, `amsmath`, `amssymb`, `booktabs`,
  `multirow`, `algorithm`, `algorithmic`, `tikz` — all standard/permitted; `\frenchspacing`
  and `\setcounter{secnumdepth}{2}` kept (required/allowed by the kit).

### 2. Undefined citation resolved (C4) — root cause: a misspelled BibTeX key

The phantom `altameem2022wildfire` turned out to be a misspelling of **Altamimi**. The real
paper was located and verified: *Altamimi et al., "Large-scale wildfire mitigation through
deep reinforcement learning," Frontiers in Forests and Global Change 5:734330 (2022),
DOI 10.3389/ffgc.2022.734330*. However, that paper is about landscape fuel-treatment
planning, **not** "grid-based fire-fighting agents" as the sentence claimed — so citing it
under the old sentence would have been a citation-claim mismatch. Fix applied:

- Added verified BibTeX entries `altamimi2022wildfire` (Frontiers, 6 authors, DOI) and
  `shen2022firehose` (Firehose: Shen & Curtis, MIT CSAIL — the grid-based Cell2Fire
  suppression benchmark this repo literally vendors in `third_party/firehose/`, provenance
  in `VENDORED.md`).
- Reworded §2: "…UAV-based detection [ghali2022, partheepan2023], landscape-scale
  fuel-treatment planning [altamimi2022wildfire], and grid-based suppression agents on
  Cell2Fire [shen2022firehose]." Every citation now matches what the cited work actually did.

### 3. Additional compile blockers found and fixed (beyond the plan)

The first clean-slate compile exposed three latent defects the review had not caught:

| Defect | Symptom | Fix |
|---|---|---|
| Missing `\affiliations{}` declaration | `! Undefined control sequence \aaai@affiliations` at `\maketitle` under `[submission]` | Added `\affiliations{Anonymous Affiliations}` |
| Duplicate `\bibliographystyle` | BibTeX error "Illegal, another \bibstyle command" (the style issues `\bibliographystyle{aaai2027}` itself at `aaai2027.sty:354`) | Removed the document-level call, comment explains why |
| `dietterich2000maxq` typed `@article` with no `journal` (it is the ICML '98 paper) | `Warning--empty journal` from BibTeX | Retyped as `@inproceedings` (fields already matched ICML '98) |

### 4. Equation and layout fixes (L2, plus one found during compile)

- Eq. (7) (`eq:pg_update`) converted from `equation` to `multline`, split before the KL
  term — no overfull box.
- Figure 1's TikZ diagram overflowed the column by 34.8pt (pre-existing; masked before by
  different fonts). `\scalebox{0.8}` → `\resizebox{\columnwidth}{!}` — guaranteed fit.

### 5. TRS wording (L3) and figure paths (L4)

- §9: "TRS $> 0.95$" → "TRS $\geq 0.95$" (interim; §9 is redesigned wholesale in Phase 5).
- Figure paths confirmed correct (`Figures/*.png` resolve; all three figures embed in the
  compiled PDF). Note the *content* of those figures is a Phase 7 matter (audit issue C7).

### 6. Reproducibility checklist (C3, skeleton stage)

All **31 answer fields** filled (the other 3 "Type your response here" occurrences are part
of the template's own instructions and must remain). Answer profile, honest to the current
paper state:

- **17 definitive answers** (yes/no/NA) where the paper already substantiates them — e.g.
  pseudocode (yes, Algorithm 1), no theoretical contributions (no + 7×NA), metrics
  formally described (yes), CIs beyond point estimates (yes), statistical tests (yes),
  runs-per-result stated (yes).
- **9 phase-tagged TODO answers** where later phases must first produce the substance:
  data-provenance appendix, release plan, ERA5/MODIS citations (→ Phase 6), hyperparameter
  ranges (→ Phase 8), code packaging, seed disclosure, compute appendix, code-comment
  cross-refs, Table 4 alignment with checkpoint configs (→ Phase 6).
- Checklist compiles standalone with zero errors.

## Success-Gate Verdict

| Gate criterion (from plan Phase 1) | Result |
|---|---|
| `pdflatex` + `bibtex` with zero errors, zero undefined citations/references | ✅ exit 0; `grep -c "^!" .log` = 0; no "undefined" in .log; no "Warning--" in .blg |
| Preamble: `[submission]` present; times/helvet/courier + manual page size gone; audited against the style's own rules | ✅ §1 above (audit documented) |
| Eq. (7) page: no overfull hbox > 1pt | ✅ zero Overfull warnings in the entire document |
| Checklist: zero template strings; every field answered or phase-tagged | ✅ 31/31 filled (17 final, 9 TODO-tagged, 5 gate answers); standalone compile clean |
| All three figures resolve in the compiled PDF | ✅ (416 KB PDF, no missing-file warnings) |

**GATE: PASS.** The desk-reject exposure named in the review's Final Verdict (blank
checklist, undefined citation, style violations) is eliminated. Remaining checklist TODOs
are owned by Phases 6 and 8 as scheduled.

## Deviations from the plan

- Three extra defects (affiliations, duplicate `\bibstyle`, dietterich entry type) were
  found and fixed beyond the plan's list — recorded in §3 above.
- The citation fix went further than "add or remove": the sentence itself misattributed the
  cited work, so the claim was corrected along with the key (integrity rule from Phase 0).

---

# Phase 2 — Baseline Rehabilitation

**Status:** ✅ COMPLETE · **Date:** 2026-07-07 · **Ran in parallel with Phase 1**
**Compute:** WSL2 Ubuntu, CPU-only (no CUDA), Python 3.12.3, venv `~/venvs/wildfire-marl`.
Training: 6 runs × 100,000 env steps, 3 parallel lanes, ≈4h13m wall-clock total
(Saudi runs ≈2.4–2.6 h each, California ≈1.6–1.7 h). Evaluation: 2 × (8 policies × 5 seeds
× 15 episodes) ≈ 1.5 h. All new artifacts isolated in `results/phase2_peer/` — the original
certified checkpoints and results in `results/runs/` and `results/phase8/` were never touched.

## Objective (from the plan)

Replace the non-credible baseline story: retrain flat MAPPO with a fair budget on the same
WEL/ISR objective it is evaluated on, give QMIX real reported results (H3), add the
communication baseline Reviewer #3 demanded (M3), evaluate everything under the certified
5-seed × 15-episode protocol with CIs, and instrument behavior (M7). Then make the
pre-committed strategy-checkpoint ruling.

## 1. Code Built (all additive; nothing certified was modified)

| Component | Location | Purpose |
|---|---|---|
| `WELISRDeltaReward` | `src/wildfire_marl/env/rewards.py` | Per-step delta of −WEL + 20·ISR; telescopes to exactly Eq. 1's macro objective — removes the objective-level confound (H1/R2 §5) |
| `CommNetActor` | `src/wildfire_marl/agents/agent_networks.py` | CommNet-style communication actor (2 rounds of mean-pooled hidden-state exchange, Sukhbaatar et al. 2016) — the M3 rebuttal baseline |
| `train_commnet` + curve logging in all trainers | `src/wildfire_marl/train/marl_train.py` | PPO trainer for the joint communication actor; every trainer now returns per-episode (return, WEL, ISR) training curves archived to CSV (H1: convergence evidence) |
| 6 training configs | `configs/marl/phase2/` | `reward: welisr`, `total_steps: 100000` (10× the original 10k), `save_dir: results/phase2_peer` |
| Evaluation harness v2 | `scripts/run_multiseed_eval_v2.py` | Certified protocol (identical seed derivation: 42+1000s, ep_seed=seed+ep), 8 policies, bootstrap CIs over seed means, Welch tests, action histograms, occupancy maps, distance-to-nearest-asset (M7); `--policies`/`--suffix` for supplementary runs |
| Curve plots | `scripts/plot_phase2_curves.py` | `results/phase2_peer/train_curves.png` + plateau statistics |

All three trainers were smoke-tested (600-step runs) before the full launch.

## 2. Protocol Validation

The v2 evaluation **reproduced the certified Table 1 per-seed values exactly** for the two
unchanged policies — Value-First Saudi (19.0, 19.0, 20.6, 19.0, 19.8 → 19.48) and Learned
Hierarchical Saudi (20.6, 19.8, 21.4, 19.8, 19.8 → 20.28), and likewise for California and
No-Op. The evaluation pipeline is deterministic; every new number below is directly
comparable to Table 1.

## 3. Results (5 seeds × 15 episodes, bootstrap 95% CIs over seed means)

### Saudi Arabia (concentrated petroleum assets) — WEL ↓ / ISR ↑

| Policy | WEL [95% CI] | ISR | Note |
|---|---|---|---|
| No-Op | 27.00 [27.0, 27.0] | 0.286 | degenerate (fire always reaches all reachable assets) |
| Greedy-Risk Heuristic | 27.00 | 0.286 | fails |
| Local Reactive (new) | 26.63 | 0.305 | treat-if-possible + random walk: fails on concentrated assets |
| **Flat MAPPO (matched, 100k)** | **27.00 [27.0, 27.0]** | 0.286 | **still exactly No-Op level** — see §5 |
| QMIX (matched, 100k) | 25.08 [24.4, 25.8] | 0.320 | learned a little (vs No-Op p=0.0086) |
| Value-First Heuristic | 19.48 [19.0, 20.1] | 0.375 | the BC teacher |
| Learned Hierarchical | 20.28 [19.8, 20.9] | 0.366 | unchanged from Table 1 |
| **CommNet (matched, 100k)** | **11.43 [9.0, 14.0]** | **0.457** | **beats everything** — vs Hierarchical p=0.0031, d=−3.72 |

### California (scattered WUI assets) — WEL ↓ / ISR ↑

| Policy | WEL [95% CI] | ISR | Note |
|---|---|---|---|
| No-Op | 25.25 [23.4, 27.1] | 0.231 | |
| QMIX (matched, 100k) | 25.25 [23.4, 27.1] | 0.231 | per-seed ≡ No-Op — see §5 |
| Greedy-Risk Heuristic | 25.21 | 0.233 | |
| Value-First Heuristic | 24.61 [23.0, 26.2] | 0.249 | |
| Learned Hierarchical | 24.53 [23.0, 26.1] | 0.251 | unchanged from Table 1 |
| Flat MAPPO (matched, 100k) | 23.33 [21.4, 25.3] | 0.249 | improved with fair training (n.s. vs No-Op, p=0.245) |
| CommNet (matched, 100k) | 19.17 [16.8, 21.5] | 0.389 | vs Hierarchical p=0.013, d=−2.09 |
| **Local Reactive (new)** | **15.13** | **0.471** | **a trivial policy beats every learned/heuristic policy** |

The Local Reactive California result (15.13) also confirms the Phase 0 M11 artifact (15.15
under the old 20-episode protocol) — that unreported baseline was real.

## 4. Convergence Evidence (H1 closed)

Training-episode WEL, first → last quintile (full curves in
`results/phase2_peer/train_curves.png`, per-episode CSVs archived):

| Run | WEL first → last quintile | Verdict |
|---|---|---|
| MAPPO Saudi | 26.96 → 27.00 | plateaued at failure (documented) |
| QMIX Saudi | 26.75 → 25.44 | marginal learning |
| CommNet Saudi | 26.80 → **10.00** | strong learning |
| MAPPO California | 25.29 → 21.01 | learned |
| QMIX California | 23.37 → 25.58 | failed to learn |
| CommNet California | 21.45 → 19.31 | learned |

The original review objection — "coordination collapse cannot be distinguished from
insufficient training budget" — is now answerable: at 10× budget on the matched objective,
independent MAPPO still plateaus at No-Op level on Saudi, with the curve to prove it.

## 5. Bit-Identity Analysis (gate requirement)

Two bit-identities with No-Op remain, and both are now **evidenced as genuine behavioral
ineffectiveness**, not evaluation bugs (action histograms in `behavior_{region}.json`):

- **Flat MAPPO Saudi ≡ No-Op** (WEL/ISR per-seed identical): the policy *acts* — 5.0% of
  its actions are `treat`, CE = 0.975 ≠ No-Op's 1.0 — but at a mean Chebyshev distance of
  6.8 cells from the nearest asset. It suppresses fire in the wrong places; asset outcomes
  never change. This is the paper's original "agents suppress fire in the wrong locations"
  claim, now with quantitative behavioral evidence (contrast: CommNet operates at distance
  2.5 and defends assets).
- **QMIX California ≡ No-Op**: near-total inactivity (0.2% treat actions) — the policy
  collapsed to not treating. Also documented.

## 6. Strategy Checkpoint (pre-committed ruling from Phase 0 §4)

**Outcome (b): fair baselines solve the task — the general "coordination collapse of flat
MARL" claim does not survive.** Specifically:

1. The claim survives **only** in narrowed form: *independent, communication-free MAPPO
   fails completely on the concentrated-asset region (Saudi) even with a 10× budget and the
   matched objective* — now a well-evidenced diagnostic finding (curves + behavior logs),
   no longer a general indictment of "flat MARL".
2. **CommNet dominates the paper's entire original result set** — better than the expert
   heuristic and the BC+KL hierarchy in both regions, significantly so. Reviewer #3's
   predicted rebuttal experiment ("a communication baseline is the natural rebuttal") was
   correct: lightweight communication repairs the failure.
3. **Local Reactive exposes region structure**: trivial coverage (treat + random walk) wins
   on scattered WUI assets (California 15.13) and fails on concentrated petroleum assets
   (Saudi 26.63). The two regions demand opposite coordination strategies — this is the
   benchmark's sharpest diagnostic property and a genuinely publishable observation.
4. The BC+KL hierarchy retains **no performance headline**: it matches its teacher and is
   beaten by CommNet everywhere and by Local Reactive on California. Its remaining value is
   interpretability and compliance-by-design, as a *reference solution*.

**Consequence for the paper (binding on Phases 3–8):** proceed as **benchmark +
fair-baseline study** (the AC's framing, now mandatory rather than optional). Claims
permitted going forward: (a) the benchmark discriminates sharply between coordination
strategies across asset topologies; (b) communication-free decentralized MAPPO fails on
concentrated assets — with evidence; (c) the hierarchy is an interpretable reference
solution with guaranteed compliance. Claims retired: "dramatically outperforms flat MARL"
(C2 — now false as a headline in *both* directions), "coordination collapse" as a general
phenomenon, and any implication that the hierarchy is the strongest solution.

## 7. Issue Disposition

| Issue | Disposition |
|---|---|
| H1 (MAPPO validity) | ✅ Retrained 100k steps, matched objective, curves archived; Saudi failure now evidence-backed, California improvement reported |
| H3 (QMIX ghost) | ✅ QMIX retrained + evaluated under full protocol in both regions; will be **reported** (hash regeneration for App. D in Phase 6) |
| M3 (communication baseline) | ✅ CommNet trained + evaluated; changes the paper's conclusions (§6) |
| M7 (behavioral analysis) | ✅ Action histograms, occupancy maps (`occupancy_*.npz`), asset-distance stats archived; Phase 7 turns these into the failure-analysis figure |
| M11 (unreported baselines) | ✅ Local Reactive evaluated under certified protocol (both regions); Greedy-Risk already in the certified set; both will be reported |
| C2 (headline claim) | Evidence ready; text change lands in Phase 8 under §6's permitted-claims list |

## 8. Success-Gate Verdict

| Gate criterion (from plan Phase 2) | Result |
|---|---|
| Per-baseline training curves showing plateau or documented divergence at disclosed budget | ✅ §4 (100k steps disclosed; curves + quintile stats archived) |
| All rows 5 seeds × 15 episodes with bootstrap CIs, incl. No-Op and all baselines; no single-run rows | ✅ §3 (`multiseed_eval_raw/aggregate_*`, `eval_summary_*.json`) |
| No bit-identity with No-Op unless action logs prove genuine behavioral collapse (evidence retained) | ✅ §5 — both remaining identities evidenced and retained for the paper |
| QMIX reported under full protocol or hashes removed | ✅ reported (§3) |
| Communication baseline under identical protocol | ✅ CommNet (§3) |
| Strategy checkpoint recorded | ✅ §6 — outcome (b), consequences binding on later phases |

**GATE: PASS.** Phase 3 (ablation rerun under the main protocol, + BC-only row) is cleared.

## Deviations from the plan

1. **Local Reactive baseline added** (not in the plan's Phase 2 list) — required to close
   M11, and it materially changed the California story. Run under the identical protocol as
   a supplementary evaluation (`*_reactive` artifacts).
2. **One training seed per algorithm/region** (seed 42), matching the original paper's
   train-once/evaluate-5-seeds design for comparability. Multi-seed *training* variance is
   not yet quantified — flagged for Phase 8's Limitations section (and Phase 3 keeps the
   same convention for ablations).
3. The evaluation env uses the certified FireSizeReward for the `total_reward` column
   (comparability with Table 1); the matched WEL/ISR objective was used for *training* the
   new baselines. WEL/ISR/CE columns are reward-independent.

---

# Phase 3 — Ablation Rerun Under the Main Protocol

**Status:** ✅ COMPLETE · **Date:** 2026-07-07 · **Compute:** single WSL2 CPU background run,
≈2h20m (7 variants: train + certified eval each). Artifacts in `results/phase3_peer/`
(checkpoints, `ablation_raw_saudi.csv`, `ablation_aggregate_saudi.csv`,
`ablation_summary_saudi.json`, `check_identity.py` mechanism verification).

## Objective (from the plan)

Make Table 3 an instrument of evidence instead of a source of contradiction: rerun every
ablation under the certified 5-seed × 15-episode protocol (C1, H8), add the missing BC-only
row (H2), and make the Full System row the same experiment as Table 1 by construction.

## 1. Design

New runner `scripts/run_ablation_v3.py` (smoke-tested before the full run):

- **Base config = the shipped checkpoint's embedded hyperparameters** (verified in Phase 0):
  40 BC episodes, 100 BC epochs, 120 RL episodes, KL 0.3, entropy 0.05, ISR weight 20,
  rl_lr 5e-5, train seed 42. Each variant removes exactly one component.
- **Full System (shipped ckpt) row = the Phase 2 certified evaluation rows** of
  `results/runs/checkpoint_hierarchical_saudi.pt` — the *same runs* as Table 1's
  Hierarchical row. The C1 contradiction is now structurally impossible.
- **Full System (retrained) control row** (beyond the plan): a fresh training run of the
  unablated system, quantifying training-run variance — the confound behind the old
  Table 3's 17.4.
- **BC-only row** (H2): Phase-1-only commander (`hierarchical_episodes: 0`).
- **w/o Target-Seeking** uses the *Phase 2 matched* MAPPO actor (100k steps, WEL/ISR
  objective) as the frozen tactical layer — a fair version of the original ablation, in
  training and evaluation.
- Evaluation: certified protocol (seeds 42+1000s, ep_seed = seed + ep, 15 episodes/seed),
  bootstrap CIs over seed means, Welch tests vs. Full System (shipped).

## 2. Results (Saudi, WEL ↓ / ISR ↑, 95% CIs over 5 seed means)

| Configuration | WEL [95% CI] | ISR | vs Full (shipped): p, d |
|---|---|---|---|
| **Full System (shipped ckpt)** — same runs as Table 1 | **20.28 [19.80, 20.92]** | 0.366 | — |
| Full System (retrained, seed 42) | 20.76 [20.28, 21.24] | 0.360 | p=0.31, d=0.69 |
| BC-only (no RL fine-tuning) | **20.28 [19.64, 20.92]** | 0.366 | **p=1.00, d=0.00** |
| w/o BC Pretraining | 27.00 [27.0, 27.0] | 0.286 | p=3.0×10⁻⁵, d=13.3 |
| w/o KL Regularization | 20.76 [20.28, 21.24] | 0.360 | p=0.31, d=0.69 |
| w/o Entropy Bonus | 20.76 [20.28, 21.24] | 0.360 | p=0.31, d=0.69 |
| w/o WEL/ISR Reward | 20.60 [19.64, 21.56] | 0.362 | p=0.64, d=0.31 |
| w/o Target-Seeking (fair MAPPO tactical) | 27.00 [27.0, 27.0] | 0.286 | p=3.0×10⁻⁵, d=13.3 |

## 3. Findings

### 3.1 C1 resolved by construction
Table 3's Full System row now *is* Table 1's Hierarchical row (same evaluation runs,
mean 20.28, identical per-seed values). The old 17.4 is explained: the retrained control
scores 20.76 under the certified 5-seed protocol — training variance is small (≈0.5 WEL,
p=0.31 vs shipped); the old value was dominated by the **single-seed evaluation** artifact,
not by training noise.

### 3.2 H2 answered: RL fine-tuning contributes nothing measurable
BC-only equals the full system exactly on both metrics (WEL 20.28 vs 20.28, p=1.00,
d=0.00; ISR 0.366 vs 0.366). Reviewer #2's hypothesis — "the fine-tuned policy may be
functionally identical to BC" — is **confirmed**. Mechanism verified directly
(`check_identity.py`): over 120 RL episodes at lr 5×10⁻⁵, commander weights move at most
~5×10⁻³ from the BC solution (and ~3×10⁻⁴ for the KL/entropy variants); greedy dispatch
decisions agree with the retrained full system on 100% (w/o KL, w/o entropy) and 93.5%
(BC-only) of 200 random sector states. The RL stage is functionally inert.

**Consequence (binding on Phase 8):** the paper's Contribution 2 ("stable training
pipeline") and §8's claims that "KL regularization prevents forgetting" (+4.6%) cannot
stand — there is nothing to forget because fine-tuning never leaves the BC solution. The
honest framing: *the system is behavior cloning of the Value-First heuristic; the
KL-anchored RL stage as configured is a no-op*, reported transparently with the mechanism
evidence. (Whether a more aggressive RL stage could exceed BC is future work — a λ/lr sweep
was considered and deferred; the current claim set does not require it.)

### 3.3 The two load-bearing components survive with fair evidence
- **w/o BC Pretraining → 27.00** (No-Op level): with sparse delayed rewards, pure REINFORCE
  from scratch learns nothing — unchanged from the old table, now with CIs.
- **w/o Target-Seeking → 27.00**: replacing the deterministic tactical layer with the
  *fairly trained* (100k-step, matched-objective) MAPPO actor still collapses the system —
  the compliance argument survives against the strongest available tactical baseline. The
  original paper made this point with the broken 10k-step actor; it now holds with a fair one.

### 3.4 H8 resolved: identical rows are real, explained, and no longer suspicious
w/o KL ≡ w/o Entropy ≡ Full (retrained) produce byte-identical per-seed WEL values
(20.6, 20.6, 21.4, 19.8, 21.4). Explanation, now evidence-backed: all three share the same
BC phase (identical demonstration seeds) and an inert RL stage, so they converge to the
same greedy dispatch policy (100% decision agreement, weight distance ≤ 3×10⁻⁴); WEL's
discrete ≈0.8-quantum lattice then makes their evaluation rows exactly equal. This is a
property of the system (inert fine-tuning), not an artifact of the protocol — and it is
precisely *why* the old single-seed table looked fabricated. The revised Table 3 will state
this mechanism in a footnote.

## 4. Issue Disposition

| Issue | Disposition |
|---|---|
| C1 (Table 1 vs Table 3) | ✅ Full System row = Table 1 runs by construction; old divergence root-caused (single-seed eval + retraining variance quantified) |
| H2 (BC-only ablation) | ✅ Added; RL contribution = 0.00 WEL (p=1.00, d=0.00), mechanism verified |
| H8 (identical rows, no CIs) | ✅ All rows have 5-seed CIs + Welch tests; remaining identities mechanistically explained (§3.4) |
| Old Table 3 values (17.4 / 18.2 ×3) | Retired — superseded by `results/phase3_peer/ablation_summary_saudi.json`, provenance TRACED |

## 5. Success-Gate Verdict

| Gate criterion (from plan Phase 3) | Result |
|---|---|
| Full System row numerically identical to Table 1's Hierarchical row (same runs) | ✅ §3.1 — reused Phase 2 certified rows; identical by construction |
| Every ablation row has 5-seed CIs and a significance test vs full system | ✅ §2 |
| BC-only row exists; RL fine-tuning contribution quantified with effect size and CI | ✅ §3.2 — Δ=0.00, d=0.00, p=1.00, mechanism verified |
| No byte-identical rows without a written, evidence-backed explanation | ✅ §3.4 — weight-distance + decision-agreement evidence archived |
| All new numbers TRACED in provenance | ✅ every row maps to `results/phase3_peer/` artifacts produced by `run_ablation_v3.py` |

**GATE: PASS.** Phase 4 (statistical reanalysis & claims recalibration) is cleared — and
inherits a major new input: the BC-equivalence finding (§3.2) must shape the permitted-claims
list alongside Phase 2's strategy ruling.

## Deviations from the plan

1. **Retrained-control row added** (not in the plan) — needed to decompose the old 17.4
   into protocol artifact vs. training variance.
2. **California ablations not rerun**: the paper's Table 3 is Saudi-only, and the plan's
   gate covers Saudi. A California ablation table remains optional for Phase 8 (the old
   unreported `ablation_results_california.csv` stays retired).
3. **w/o Target-Seeking upgraded** to the fair Phase 2 MAPPO actor rather than the original
   broken 10k-step actor (integrity rule: no ablation against a known-broken component).

---

# Phase 4 — Statistical Reanalysis & Claims Recalibration

**Status:** ✅ COMPLETE · **Date:** 2026-07-07 · **Compute:** one background run ≈20 min
(150 instrumented hierarchical episodes across both regions + statistics from existing
Phase 2/3 aggregates). Artifacts in `results/phase4_peer/`:
`compliance_reassignment_{region}.json`, `tost_results.json`, `stats_tables.json`,
`reward_decomposition.json`, and the frozen **`claims_evidence.md`**.
New script: `scripts/run_phase4_stats.py`.

## Objective (from the plan)

Replace every statistically indefensible inference with a defensible one: TOST equivalence
testing (H7), a uniform test protocol with stated n/df (M8), degenerate-variance caveats,
region-accurate headline numbers (C2 analysis), the reassignment rate (M2), the reward
decomposition (L1) — and freeze the abstract-eligible claims list that binds Phase 8.
Also: recompute and archive the Table 2 compliance metrics (M12 from Phase 0).

## 1. Equivalence Testing (H7) — TOST, pre-registered ±5% margin

| Comparison | mean diff (WEL) | margin | p_TOST | Equivalent at α=0.05? |
|---|---|---|---|---|
| Hierarchical vs Value-First (Saudi) | +0.80 | ±0.974 | 0.355 | **NO** |
| Hierarchical vs Value-First (California) | −0.08 | ±1.231 | 0.200 | **NO** |
| BC-only vs Full System (Saudi) | 0.00 | ±1.014 | 0.044 | **YES** |

The paper's "statistical parity" claim (Finding 2) is now formally dead: equivalence with
the teacher is **not established in either region** — on Saudi the heuristic retains a
nominal 0.8-WEL edge (d=1.12), and on California n=5 is simply underpowered for a ±5%
margin despite a near-zero difference. Permitted wording is frozen in `claims_evidence.md`
(P4): "comparable; equivalence not established." Conversely, the Phase 3 BC-equivalence
finding is *positively* supported: BC-only is TOST-equivalent to the full system (P5) —
the strongest statistical statement in the revised paper.

## 2. Table 2 Recomputed and Archived (M12) — old values not reproducible

Certified protocol (5 seeds × 15 episodes, shipped checkpoint), archived with dispersion:

| Metric | Paper (old, unarchived) | Recomputed Saudi | Recomputed California |
|---|---|---|---|
| Target compliance rate | 1.000 / 1.000 | **1.000 ± 0.000** | **1.000 ± 0.000** |
| Sector drift (cells) | 4.32 / 4.18 | **1.99 ± 0.35** | **2.47 ± 1.06** |
| Dispatch latency (steps) | 1.67 / 1.72 | **0.98 ± 0.29** | **1.21 ± 0.62** |

Compliance = 1.000 is confirmed (it is guaranteed by construction). The old drift and
latency values — which existed only in un-archived stdout (Phase 0, M12) — are **not
reproducible** under the certified protocol and are superseded. This is a fourth instance
of the Phase 0 pattern (paper numbers not generated from certified artifacts) and is now
structurally fixed: the new values live in versioned JSON with the generating script.

## 3. Reassignment Rates (M2) — Reviewer #1's Question 2, answered with data

| Metric | Saudi | California |
|---|---|---|
| Cross-sector dispatch rate (assigned ≠ current sector) | **27.1%** | **36.6%** |
| Assignment-change rate (vs previous macro-step) | **19.2%** | **23.6%** |

The reviewers' suspicion is *partially* confirmed: roughly two-thirds to three-quarters of
dispatches keep an agent in its current sector. Strategic dispatch is therefore honestly
described as **initial placement plus periodic reallocation** — binding wording (P8) for
Phase 8's §7 and for the Phase 7 regeneration of Figure 3 (which should show a real
reallocation event and cite these rates in the caption).

## 4. Uniform Test Tables (M8) and Degenerate-Variance Policy

`stats_tables.json` recomputes every comparison (all Phase 2 policies vs No-Op and vs
Hierarchical, both regions and both metrics; all Phase 3 ablations vs the full system)
under one protocol: Welch's t with **Welch–Satterthwaite df reported per test**, pooled-SD
Cohen's d **suppressed when both sides are degenerate and flagged when one side is**,
bootstrap 95% CI on every mean difference, documented sign convention. Headline example
now available for the paper: CommNet vs Hierarchical (Saudi) diff = −8.85 WEL,
CI [−11.3, −6.1], t=−5.89, df=4.4, p=0.0031, d=−3.72. The old d=13.28-vs-zero-variance
presentation is permitted only with the explicit degenerate caveat (P6).

## 5. Reward Decomposition (L1)

`total_reward` is dominated by a policy-independent burn constant: No-Op baselines −549.1
(Saudi) / −423.1 (California); policy effects range from +0.15 to +28.5, i.e. **0.03–6.4%
of the baseline magnitude**. Phase 8 ruling (P9): drop the raw reward column or report
Δreward vs No-Op with CIs — bolding raw −548.07 vs −548.13 (0.01% difference) is retired.

## 6. Claims-to-Evidence Freeze

`results/phase4_peer/claims_evidence.md` locks: **9 permitted claims** (P1–P9, each with
its maximum permissible strength and artifact pointer) and **9 retired claims** (including
"dramatically outperforms flat MARL", "statistical parity", general "coordination
collapse", "KL prevents forgetting", the vacuous TRS claim, and the superseded Table 2/4
values). Phase 8's rewrite may not exceed this list; Phase 5–7 outputs append to it but
cannot relax it.

## 7. Success-Gate Verdict

| Gate criterion (from plan Phase 4) | Result |
|---|---|
| Every paper-destined statistical claim maps to a test with stated n, df, test type, margin | ✅ `stats_tables.json` (n_a/n_b, t, df, p, d, CI per test) + `claims_evidence.md` |
| TOST results for both regions; permissible "matches the expert" wording frozen | ✅ §1; P4/P5 wording locked |
| No effect size on (near-)degenerate distributions without caveat | ✅ suppression/flagging built into the machinery; P6 rule |
| Reassignment-rate statistics computed for both regions | ✅ §3 |
| Abstract-eligible claim set explicitly listed for Phase 8 | ✅ §6 (P1–P9 + retired list) |

**GATE: PASS.** Phase 5 (generalization & transfer redesign) is cleared.

## Deviations from the plan

1. **BC-only TOST added** (not in the plan's list) — the Phase 3 finding deserved a
   positive equivalence statement, and it is the one comparison where equivalence holds.
2. **Table 2 discrepancy discovered** (drift/latency ≈2× off): the plan expected a simple
   re-archival; the recomputation instead superseded the paper's values — recorded as an
   extension of the Phase 0 integrity pattern and folded into the retired-claims list.

---

# Phase 5 — Generalization & Transfer Redesign

**Status:** ✅ COMPLETE · **Date:** 2026-07-07 · **Compute:** one background run, 3 parallel
lanes, ≈2h. Artifacts in `results/phase5_peer/`: `generalization_{raw,aggregate}_{region}.csv`,
`generalization_summary_{region}.json`, `scale_{raw,aggregate}_{region}.csv`,
`scale_summary_{region}.json`, `generalization_dwel.png`, `envmods/` (the perturbed
landscapes). New scripts: `scripts/run_generalization_v2.py`, `scripts/run_scale_study.py`,
`scripts/plot_generalization.py`; `run_multiseed_eval_v2.run_episode` gained a
`reset_options` passthrough.

## Objective (from the plan)

Replace the unfalsifiable TRS metric with generalization evidence that can fail (M1, H9),
widen the evaluation axis to answer the "toy scale, one setting" objection (M6), and settle
the fire-cadence framing (M4).

## 1. New Metric: ΔWEL vs same-condition No-Op (falsifiability built in)

For every held-out condition, No-Op is **re-run in that condition** and each policy is
scored as ΔWEL = WEL(No-Op) − WEL(policy). No-Op therefore scores exactly 0 by
construction; a policy that fails to help scores 0 (or negative). This is the property the
old TRS lacked — TRS ≈ reward/reward ≈ 1.0 for any policy including No-Op, so it could not
express failure.

**Falsifiability demonstrated (the gate requirement):** several strategic policies score
ΔWEL ≈ 0 in real conditions — the metric visibly fails when generalization fails (see §2).

## 2. Held-out Generalization Results (5 seeds × 15 episodes, both regions)

ΔWEL vs same-condition No-Op (higher = adds value; **0 = no better than inaction**):

| Condition | Region | Value-First | Local Reactive | Hierarchy | CommNet |
|---|---|---|---|---|---|
| Held-out ignitions | Saudi | +7.20 | +1.47 | +7.20 | **+17.60** |
| Held-out ignitions | California | +0.00 | +7.83 | +0.00 | +5.40 |
| Wind +90° | Saudi | +11.44 | +1.19 | +11.28 | **+24.57** |
| Wind +90° | California | +0.00 | +6.32 | +0.00 | +4.44 |
| **Rotated assets** | Saudi | +0.08 | +2.76 | **+0.00** | −0.03 |
| **Rotated assets** | California | +0.12 | +10.16 | **+0.04** | +7.25 |
| **Cross-region transfer** | Saudi | — | — | **+0.00** | +0.43 |
| **Cross-region transfer** | California | — | — | **+0.04** | +0.08 |

Three genuine, publishable negative findings the old metric masked:

- **Rotated-asset collapse (P10).** When the infrastructure layout is rotated 90° (unseen
  arrangement, unchanged fuel/terrain/weather physics), the learned hierarchy and its
  Value-First teacher fall to **No-Op level** (Saudi ΔWEL +0.00/+0.08; California
  +0.04/+0.12). The egocentric policies — Local Reactive and CommNet, which read local
  infrastructure through observation channels — still adapt (Local Reactive +2.76/+10.16).
  The sector-dispatch policies do not generalize to novel asset geometry.
- **Cross-region transfer is absent (P11).** The *corrected* transfer experiment (the
  other region's trained policy on this region's standard env) yields ΔWEL ≤ +0.43 over
  No-Op in both directions. This directly falsifies the paper's "strong cross-region
  generalization (TRS > 0.95)" — transfer neither helps nor is robust; the old ≈0.95 was an
  artifact of TRS being a ratio of a near-constant reward.
- **California hierarchy is brittle under ignition/wind shift** (ΔWEL +0.00): it only ever
  matched No-Op on California to begin with (Phase 2), and any distribution shift removes
  even that.

CommNet is the consistent generalizer (positive ΔWEL in 7 of 8 held-out cells, strongly so
on Saudi), reinforcing the Phase 2 strategy ruling.

## 3. Scale Study — Agent Count (M6)

WEL by team size (certified protocol; CommNet is the N=3 checkpoint run **zero-shot** at
larger N via its size-agnostic mean-pooled communication):

| Region | Policy | N=3 | N=6 | N=10 |
|---|---|---|---|---|
| Saudi | CommNet (zero-shot) | 11.43 | 12.25 | 9.59 |
| Saudi | Local Reactive | 26.63 | 25.80 | 24.69 |
| Saudi | Value-First | 19.48 | 19.48 | 19.48 |
| California | CommNet (zero-shot) | 19.17 | 16.60 | 19.39 |
| California | Local Reactive | 15.13 | 8.81 | 5.16 |
| California | Value-First | 24.61 | 24.53 | 24.49 |

- CommNet **holds zero-shot** across team sizes (no collapse at N=10) — evidence its
  communication mechanism scales.
- Local Reactive **improves monotonically with team size** on scattered California assets
  (15.1→8.8→5.2): more agents = more coverage, exactly as expected for the reactive regime.
- Value-First is flat (fixed sector logic, indifferent to extra agents).
- The **learned hierarchy is architecturally locked to N=3** (per-agent commander heads),
  so it is excluded above N=3 and reported as a scaling limitation (P12).

**Grid-size axis (64²) documented infeasible this pass:** the Cell2Fire landscape
conversion and infrastructure rasters exist only at 32² (`data/*/grids/64x64/` holds the
raw tensors but no `asset_type/criticality/blast_radius` layers or Cell2Fire `Forest.asc`),
and every policy would require retraining. Recorded as a stated limitation rather than a
result — the plan's gate explicitly permits a documented infeasibility note.

## 4. Fire-Cadence Decision (M4) — relabel, do not change dynamics

Made from the Phase 0 §2.5 code evidence: `steps_per_action=60` means **60 fire periods
(simulated minutes) per agent step**, so the fire advances **every** step — 150 fire
advances per 150-step episode, not ≤2. There is no slower cadence in play to "speed up";
the hazard already evolves maximally per step. Therefore the "dynamically evolving hazard
landscape" framing is **correct and retained**; only Table 4's mislabeled row
("Steps per fire update = 60") is fixed in Phase 6/8 to "Simulated minutes per agent step:
60 (fire advances every step)". No dynamics change, no rerun needed — the ≤2-updates
inference was a labeling artifact, not a physics fact.

## 5. Figure 4 Decision — retire TRS heatmap, replace with ΔWEL bars

The old `transfer_heatmap.png` (all-green TRS ≥ 0.95, Phase 0 issue C7/H9) is **retired**.
Replacement drafted: `results/phase5_peer/generalization_dwel.png` — grouped ΔWEL bars per
policy per condition, both regions, with the 0-line making the rotated-asset / cross-region
collapse visible. Phase 7 finalizes styling and caption; the data and message are locked.

## 6. Success-Gate Verdict

| Gate criterion (from plan Phase 5) | Result |
|---|---|
| Generalization metric assigns a clearly bad score to No-Op (falsifiability in-paper) | ✅ §1 — No-Op = 0 by construction; strategic policies also hit 0 under rotated/cross-region (§2), proving the metric can fail |
| Held-out ignition/wind/layout results, both regions, standard protocol | ✅ §2 (4 conditions × 2 regions × 5 policies × 5 seeds × 15 episodes) |
| Scale-study results for ≥2 agent counts and ≥2 grid sizes, or documented infeasibility | ✅ §3 — N∈{3,6,10}; 64² infeasibility documented |
| Fire-cadence decision made from measured evidence and recorded; hazard language matches sim | ✅ §4 — relabel decision from code evidence; framing retained |
| Figure 4 regenerated or retired | ✅ §5 — TRS retired, ΔWEL replacement drafted |

**GATE: PASS.** Phase 6 (environment & data transparency) is cleared. `claims_evidence.md`
extended with P10–P12 and two new retired claims (cross-region "strong transfer",
"generalizes across layouts").

## Deviations from the plan

1. **Cross-region transfer folded into the generalization suite** (rather than a separate
   TRS-renormalization) — the plan offered "replace OR renormalize TRS"; replacement is
   cleaner and the corrected experiment doubles as the H9 fix (the old §9 run was broken,
   not merely insensitive).
2. **64² deferred with documentation** rather than run — infrastructure rasters do not
   exist at 64² and building them + retraining is out of scope for this remediation pass;
   the gate explicitly allows a documented infeasibility note.
3. **CommNet added to the generalization/scale suites** (beyond the plan's hierarchy-focused
   wording) — required because Phase 2 made CommNet the strongest policy, so its
   generalization is now the load-bearing question.
