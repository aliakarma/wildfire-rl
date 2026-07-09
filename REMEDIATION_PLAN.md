# Week #6 → AAAI Main-Track Submission: Phase-by-Phase Plan

## Context

**Why this work is being done.** The repo is a mature MARL wildfire-suppression project
(Cell2Fire physics, Saudi petroleum + California WUI landscapes, MAPPO/QMIX/CommNet baselines,
a hierarchical commander, full reproducibility stack). The current AAAI draft
(`AAAI Template/AnonymousSubmission2027.tex`) is honestly framed as a **"benchmark + diagnosis
with a negative result"**: it states *"we make no claim of algorithmic novelty,"* the proposed
hierarchy's *"RL stage is inert"* (it merely behavior-clones the Value-First heuristic), it
**loses to CommNet on Saudi and to a trivial reactive policy on California**, and *"cross-region
transfer is essentially absent."*

**The root cause, from the current results** (`results/phase2_peer/eval_summary_saudi.json`,
`results/runs/saudi_ablation_table.tex`): the fire burns **~884/1024 cells (~86%) regardless of
policy**. Flat MAPPO produces WEL 27.0 / ISR 0.286 — *byte-identical to No-Op*. The learned
hierarchy (20.3 WEL) is **beaten by the Value-First heuristic** (19.5, p=0.12). With
`steps_per_action=60` (60 simulated minutes of fire growth between agent actions) and 3 agents
treating 1 cell each, suppression is physically impossible — so **no method can demonstrate
skill**. This is exactly why the proposed model ties No-Op.

**The goal (user's explicit intent).** Turn this into a submission where **our proposed model
decisively beats every baseline**, complete all Week-6 tasks, and reach AAAI main-track–ready
state. This requires two coupled fixes: (1) put the environment in a *suppression-relevant
regime* where good decisions matter (Task 4's "less spreading" is the linchpin), and (2) build a
proposed method that *genuinely learns* to win (not an inert BC clone).

**Locked decisions (from clarification):**
- **Proposed model = Hierarchy + Communication**: strategic commander (macro sector dispatch) +
  a *learned, communicating* tactical layer (CommNet-style message passing), trained end-to-end.
  This is the paper's own stated "clearest path to a hierarchy that beats its teacher."
- **Fire regime = recalibrate default to suppression-relevant + add easy/medium/hard regimes** as
  a robustness study.
- **Compute = hybrid**: local CPU/GPU for development + smoke tests; Colab GPU for final training,
  larger budgets, and multi-seed evaluation.

### Scientific-integrity guardrails (apply to every phase)
- The regime recalibration is **defined and documented as a benchmark revision** (a suppression
  regime where the decision problem is non-trivial *and* solvable), never tuned per-method or
  per-seed to flatter results. One env config drives all methods.
- Keep the honest **benchmark/diagnostic contributions**; add the winning method *on top*, do not
  delete the diagnosis. Preserve disjoint train/eval ignition seed streams
  (`SCENARIO_SEED_OFFSET`), bootstrap CIs, Welch tests, TOST, effect sizes.
- **Go/no-go fallback (Phase 3 gate):** if the proposed model cannot beat the heuristic + CommNet
  after honest tuning, we do **not** fabricate — we fall back to the strengthened benchmark paper
  and report the method's regime-dependent wins truthfully. This preserves the project's existing
  Phase-8 go/no-go philosophy.

### Week-6 task → phase mapping
| Task | Phase |
|---|---|
| Obtain/train stronger policy | 2, 3 |
| Visualize rollout GIFs for ablations | 5 |
| Produce results that showcase findings | 4 |
| More frequent + more random spawns, less spreading | 1 |
| Cross-environment validation (Saudi↔Cali) + failure modes | 6 |
| Paper: AAAI-style framework figure | 7 |
| Overlay real OSM (Saudi & Cali) — low priority | 8 |

---

## Phase 0 — Reproduce baseline & stand up the hybrid pipeline

**Goal.** Lock a trustworthy starting point and the local-dev / Colab-final workflow before
changing anything.

**Work.**
- Reproduce current eval locally (WSL2 venv) and on Colab: confirm ~86% burn and
  MAPPO≈No-Op using existing `scripts/run_multiseed_eval_v2.py` + `results/phase2_peer/`.
- Create a Colab training notebook that builds the Cell2Fire binary
  (`third_party/firehose/cell2fire/Cell2FireC`, `make -f Makefile_UBUNTU`), installs
  `pip install -e ".[dev]"`, and runs a smoke train→eval→table cycle end-to-end.
- Establish a "smoke" tier (tiny `total_steps`, 1 seed, local) vs "full" tier (Colab GPU) as
  reusable configs, mirroring the existing `results/phase2_peer/smoke/` pattern.

**Success criteria (gate).**
1. Current headline numbers reproduced within noise locally **and** on Colab.
2. Colab notebook runs a full smoke `train → eval → emit table` for one method without manual
   intervention; artifacts land in `results/`.
3. Cell2Fire binary builds cleanly in Colab. No code behavior changed yet.

---

## Phase 1 — Environment redesign: less spreading + more/random spawns (Task 4)

**Goal.** Move the benchmark into a *suppression-relevant* regime with a richer ignition
distribution, so policy skill produces large, honest separation. **This is the linchpin phase —
nothing downstream can show "much better results" until this gate passes.**

**Work — 1A: spread recalibration (the "less spreading" lever).**
- Expose the spread/capacity knobs already latent in the code as a single **regime config**:
  `steps_per_action` (dominant lever — cut from 60 toward ~5–15), `steps_before_sim` (fire head
  start), `ros_cv`, wind rows in `Weather.csv`, `action_diameter` (1→2 treatment patch), and
  `num_agents`. Files: `env/single_agent_env.py`, `env/cell2fire_binding.py`
  (`--Fire-Period-Length`, `--ROS-CV`), `env/marl_env.py`.
- Define four named regimes in `configs/marl/regimes/{easy,medium,hard,default}.yaml`. **default =
  suppression-relevant** (calibrated so No-Op loses meaningful infrastructure but a good policy can
  save most of it). Calibrate by sweeping the knobs and reading No-Op vs Value-First separation —
  *not* by looking at the proposed model.

**Work — 1B: ignition redesign ("more frequent, more random").**
- Support **multiple simultaneous ignitions**: extend `_write_ignition_csv`
  (`env/single_agent_env.py`) and `write_ignitions_csv` (`data/ignition.py`) to emit multiple
  `Ncell` rows; sample K ignition cells per episode.
- Add **stochastic ongoing spawns**: a wrapper-level re-ignition step that injects new fires with
  a per-step probability (model it on the existing `infra/cascade.py::cascade_step` injection
  path so it stays in the wrapper, never in Cell2Fire physics).
- Add a **"more random" mode**: uniform ignition sampling over fuel cells as an alternative to the
  FIRMS detection-weighted `IgnitionSampler`. Preserve the leakage-free disjoint train/eval seed
  streams (`sample_train`/`sample_eval`, offset `100000`).

**Success criteria (gate).**
1. In `default` regime: No-Op ISR has clear headroom (assets are *savable*, not near-total loss),
   and **Value-First beats No-Op on WEL and ISR with a large effect (non-overlapping bootstrap
   CIs, Cohen's d > 1)** — proving the env now rewards skill.
2. Multi-ignition (K≥2 or stochastic spawns) is active, reproducible, and leakage-free across
   disjoint train/eval seeds (verify with `reproducibility/check_seed_integrity.py`).
3. easy/medium/hard produce **monotonic** difficulty (No-Op WEL increases with difficulty).
4. Existing `tests/` still pass; regime is one config, identical across methods.

---

## Phase 2 — Build the proposed model: Hierarchy + Communication (Task 1a)

**Goal.** Replace the inert BC-clone hierarchy with a genuinely trainable communicating
hierarchical policy that can *beat* the heuristic.

**Work.**
- New tactical actor `CommTacticalActor` in `agents/agent_networks.py`: fuses the CommNet-style
  mean-pooled message passing (already in `CommNetActor`) with the strategic sector target as a
  conditioning input, and is **learned** (replaces the deterministic Chebyshev target-seeking that
  made the RL stage inert).
- Keep the strategic commander (`agents/strategic_controller.py`) for macro sector dispatch, but
  train commander + comms-tactical **end-to-end** (or commander-RL over a learned-PPO tactical
  layer with communication). New trainer: extend `train/hierarchical_train.py` or add
  `train/hier_comm_train.py`, reusing the MAPPO GAE/PPO machinery in `train/marl_train.py`.
- Remove the N=3 architectural lock (paper's stated limitation): shared-parameter agent heads so
  the model scales to N∈{3,6,10}.
- Objective: keep the WEL/ISR macro reward (Eq. 1); add tactical shaping (frontier/asset proximity)
  to fix the sparse-reward credit-assignment problem the paper flags.

**Success criteria (gate).**
1. New model **trains stably** on smoke tier (return/ISR curves rise, not flat).
2. On a **single seed** in the `default` regime it **beats both No-Op and the Value-First
   heuristic** on WEL and ISR (strictly better, not tied). If it can't clear the heuristic on one
   seed, iterate the architecture/reward here before spending Colab budget on full sweeps.

---

## Phase 3 — Train the stronger policy + tune (Task 1b) — **the "much better results" gate**

**Goal.** Produce the final, strong checkpoints and confirm the proposed model wins across seeds.

**Work.**
- Full Colab-GPU training: longer budgets, tuned hyperparameters (lr, entropy, comm rounds,
  KL/λ, `T_macro`, tactical shaping weights); optional **curriculum** (easy→hard).
- Retrain **all methods on the identical new env** for a fair comparison — No-Op, Local Reactive,
  Greedy-Risk, Value-First, MAPPO, QMIX, CommNet, and **Hierarchy+Comms** — for **both regions**.
- Archive checkpoints + `train_curve_*.csv` (existing convention), refresh SHA-256 hashes.

**Success criteria (gate — go/no-go).**
- Across **≥5 seeds** in `default`, the proposed **Hierarchy+Comms achieves best (or tied-best)
  WEL and ISR in both regions**, and **beats CommNet and Value-First with statistical
  significance** (Welch t on seed means, p<0.05, meaningful Cohen's d) in at least the
  concentrated-asset region (Saudi), and is no worse than best elsewhere.
- If the full win is not achieved after honest tuning: invoke the documented fallback (strengthened
  benchmark paper + truthful regime-dependent wins) rather than forcing numbers. **Do not proceed
  to paper repositioning until this gate is explicitly evaluated.**

---

## Phase 4 — Multi-seed results tables & figures (Task 3)

**Goal.** Regenerate every quantitative artifact from the fresh runs.

**Work.**
- Full 5-seed × 15-episode protocol, all methods × both regions × `default`, plus the
  easy/medium/hard **robustness table**. Reuse `scripts/run_multiseed_eval_v2.py`,
  `eval/statistics.py`, `eval/significance.py`, `scripts/build_report_tables.py`.
- Regenerate main results table, statistical tables (bootstrap CIs, Welch, TOST, effect sizes),
  baseline training curves.
- **Component ablations of the proposed model**: w/o communication, w/o hierarchy (flat comms),
  w/o learned tactical (back to Chebyshev), w/o RL fine-tune, w/o tactical shaping — showing each
  piece contributes.

**Success criteria.**
1. Main + ablation + significance tables regenerated **only** from archived CSVs (traceable).
2. Proposed model is best-in-class per the Phase-3 gate; **ablations show comms and hierarchy each
   add measurable, significant value**.
3. Robustness table shows the ranking holds (or degrades gracefully) across regimes.

---

## Phase 5 — Rollout GIFs for ablations (Task 2)

**Goal.** Make the win visible.

**Work.**
- Render rollout GIFs for each policy/ablation × both regions with the new checkpoints via
  `scripts/render_rollout.py` + `viz/rollout.py` (already supports agents, targets, WEL/ISR/CE
  overlays, sector grid). Add the new policy name and any comms-message overlay.
- Curate a side-by-side (proposed vs No-Op vs CommNet) that showcases coordinated firebreaks /
  asset protection.

**Success criteria.**
1. GIFs for all key policies × both regions render cleanly (replace `results/phase13/*.gif`).
2. The proposed model's GIF **visibly protects more infrastructure** (fewer burned assets,
   coordinated suppression) than baselines; on-frame WEL/ISR/CE match the Phase-4 tables.

---

## Phase 6 — Cross-environment validation + failure modes (Task 6)

**Goal.** Saudi↔California transfer with the new policies, and an honest failure-mode analysis.

**Work.**
- Train-on-A → eval-on-B both directions; compute TRS / cross-domain gap / asymmetry
  (`eval/transfer.py`) and the held-out generalization sweep (novel ignitions, wind 90°, rotated
  layout, cross-region) via `scripts/run_transfer_v2.py` + `scripts/run_generalization_v2.py`.
- Multi-ignition training (Phase 1) should *improve* transfer vs the paper's "essentially absent";
  characterize where/why it still breaks (e.g., rotated asset geometry).
- Transfer GIFs via `scripts/render_transfer.py` + `viz/transfer.py`.

**Success criteria.**
1. Full transfer matrix (both directions) + generalization figure regenerated with new policies.
2. **Failure modes documented with evidence** (which conditions break which policy class, and why).
3. Honest characterization of whether the proposed model transfers better than baselines.

---

## Phase 7 — Paper: framework figure + reposition & rewrite (Task 7)

**Goal.** AAAI-style framework figure and a coherent, honest paper where the method wins.

**Work.**
- Study 2–3 recent AAAI framework/architecture figures for tone/style; build a polished
  `Figures/architecture.*` for Hierarchy+Comms (upgrade the current TikZ Fig. 1).
- Rewrite `AAAI Template/AnonymousSubmission2027.tex`: reposition from "diagnosis where the method
  ties" to "benchmark **+ a method that wins**," while retaining the diagnostic contribution.
  Update abstract, contributions, Results (Findings), Ablation, Generalization, Limitations, and
  all tables/figures/hashes. Use the `bibtex-validator` skill on `aaai2027.bib`.

**Success criteria.**
1. Paper **compiles**; new framework figure present in AAAI style.
2. Every number/claim matches Phase-4/6 artifacts; contributions now include the winning method;
   limitations remain honest.

---

## Phase 8 — OSM overlay for Saudi & California (Task 5, low priority)

**Goal.** Overlay real OpenStreetMap context onto the grids for figures.

**Work.**
- Use existing `data/saudi_eastern_province/raw/osm` + `figures/*_satellite_bg.png`; render OSM/
  asset overlays aligned to the ROIs from Appendix G. Add a small viz helper; integrate into paper
  landscape figure and README where useful.

**Success criteria.**
- OSM overlay figures for both regions produced and integrated. **Explicitly deferrable** if
  Phases 1–7 consume the time budget.

---

## Phase 9 — AAAI submission readiness

**Goal.** Reproducibility, integrity, and format sign-off.

**Work.**
- Refresh reproducibility manifest + hashes (`reproducibility/make_manifest.py`,
  `reproducibility/check_seed_integrity.py`); regenerate the reproducibility certificate
  (`results/phase15/`); complete `AAAI Template/ReproducibilityChecklist.tex`.
- Integrity pass: confirm the regime revision is documented, no per-method/per-seed tuning, no
  cherry-picking; limitations updated.
- Format: page limit, anonymization, reference validity (`bibtex-validator`), final proofread
  (optionally `humanize-latex` / `peer-review` skills).

**Success criteria.**
1. Reproducibility certificate regenerated; checklist complete; a fresh Colab clone reproduces the
   headline numbers.
2. Paper within page limit, anonymized, references valid. **Project ready to submit to AAAI main
   track.**

---

## Cross-cutting: compute workflow & risk

- **Hybrid loop:** iterate architecture/regime locally on the smoke tier (WSL2 venv per
  `~/venvs/wildfire-marl`), then launch full seeds/budgets on Colab GPU; pull artifacts back into
  `results/`. Keep the smoke/full config split from Phase 0.
- **Primary risk:** Phase-3 gate not met. Mitigations: tactical reward shaping, curriculum,
  comms-round/KL tuning; and the documented benchmark-paper fallback so the submission is never
  blocked on an unproven claim.
- **Commits:** per project convention, the user commits/pushes himself mid-session — do not commit
  unless asked.

## Verification (end-to-end)

1. **Env gate (Phase 1):** run No-Op vs Value-First multi-seed eval in `default`; confirm
   large separation + reproducible multi-ignition; `pytest tests/`.
2. **Method gate (Phases 2–3):** train Hierarchy+Comms; confirm single-seed win, then ≥5-seed
   significant win vs CommNet + Value-First via `eval/significance.py`.
3. **Artifacts (Phases 4–6):** regenerate tables/GIFs/transfer strictly from archived CSVs;
   spot-check that on-frame GIF metrics equal the table values.
4. **Paper (Phases 7,9):** LaTeX compiles; reproducibility certificate + checklist pass; fresh
   Colab clone reproduces headline numbers.
