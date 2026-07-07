# PEER_REVIEW_REMEDIATION_PLAN — Fixing All Issues from the AAAI 2027 Peer Review Report

**Source review:** `AAAI Template/Peer_Review_Report.md` (3 reviewers + Area Chair, Q1/A* calibration)
**Manuscript under revision:** `AAAI Template/AnonymousSubmission2027.tex` + `ReproducibilityChecklist.tex` + figures
**Verdict being remediated:** Reject in current form (≤ 5% acceptance estimate, desk-reject exposure); target after remediation: 35–50% at AAAI 2028 / AAMAS main track, higher at AI-for-Social-Impact track.

**Governing principle (from the Area Chair):** the concerns require *new experiments, not rewording*. This plan therefore sequences experimental rehabilitation **before** any prose rewriting, and every phase ends with a hard success gate that must pass before the next phase begins. Execution of each phase should be logged in `Phases.md` following the existing convention (actions, deviations, evidence, gate result).

**Execution environment:** all experimental phases (2, 3, 5, and parts of 4/7) run in the WSL2 Ubuntu environment used for V2 simulation work. LaTeX/writing phases (1, 6, 8, 9) can run on Windows.

---

## 1. Consolidated Issue Register

Every issue raised by any reviewer, deduplicated, with a stable ID used throughout this plan. Severity follows the review's own Revision Roadmap, with reviewer-specific additions folded in.

### Critical — desk-reject or trust-destroying (must fix, no exceptions)

| ID | Issue | Where raised |
|----|-------|--------------|
| **C1** | Table 1 vs. Table 3 contradiction: full system on Saudi reported as WEL 20.28 / ISR 0.366 (Table 1) and WEL 17.4 / ISR 0.400 (Table 3) with no explanation | R1-W1, R2 §6.3, AC-1 |
| **C2** | Abstract/Finding 3 claim "dramatically outperforms flat MARL" is Saudi-only; Appendix C shows California is non-significant (p = 0.620, d = −0.327) | R1-W2, R2 §7, AC-2 |
| **C3** | `ReproducibilityChecklist.tex` entirely blank — 34 "Type your response here" fields (verified) | R1-W4, AC-5 |
| **C4** | Undefined citation `altameem2022wildfire` — cited in §2, absent from `aaai2027.bib` | R1-W3, R2 §7 |
| **C5** | LaTeX preamble violations: missing `[submission]` option; `times`/`helvet`/`courier` loaded against `aaai2027.sty` prohibitions; manual `\pdfpagewidth`/`\pdfpageheight`; `caption` package and `\setcounter{secnumdepth}{2}` unverified against author-kit forbidden list; `\pdfinfo` a no-op without `[submission]` | Roadmap, Final Verdict |

### High — empirical core

| ID | Issue | Where raised |
|----|-------|--------------|
| **H1** | Flat MAPPO baseline non-credible: 10,000 training steps, results bit-identical to No-Op in both regions, no CIs, no training curves, reward it was trained on unspecified | R1-W6, R2 §5, AC-3 |
| **H2** | Missing BC-only (Phase-1-only) ablation — RL fine-tuning's contribution unestablished; "the single most important missing ablation" | R2 §3.2, AC-4 |
| **H3** | QMIX ghost results: Appendix D lists QMIX checkpoint hashes but zero QMIX results in the paper | R1 §6, R3 §9.1 |
| **H4** | Figure 2 caption says "Flat MARL" but left panel is titled "No-Op Rollout (Saudi)"; both panels show identical footer metrics; WEL 4.0 unexplained vs. Table 1 scale | R1 §6, R2 §6.4 |
| **H5** | GIS data provenance entirely absent: no source, resolution, preprocessing-to-32×32 method, license, or release plan for the "real-world GIS-derived topographies" | R1 §9, R3 §9.4 |
| **H6** | No compute/hardware disclosure: GPU/CPU, memory, wall-clock, library versions, seed values all absent | R1-W8, Roadmap |
| **H7** | "Statistical parity" argued from non-significance at n = 5 (absence-of-evidence fallacy; Saudi d = −1.23 is large) — needs TOST/equivalence bounds | R1 §8, Roadmap |
| **H8** | Ablation table anomalies: no seeds/CIs/tests; three unrelated ablations byte-identical (18.2/0.390); entropy ablation identical to full system | R1-M3, R2 §6.3 |

### Medium — claims, metrics, and framing

| ID | Issue | Where raised |
|----|-------|--------------|
| **M1** | TRS transfer metric cannot fail: rewards dominated by ≈ −550/−421 constant, so any policy (incl. No-Op) scores TRS ≈ 0.95–1.00; Figure 4 uninformative | R1 §10, R2 §6.1, R3 §7 |
| **M2** | Figure 3 contradicts caption (dispatch stars at agent positions, agents adjacent, not "distributed to distinct sectors"); 1.67-step dispatch latency implies commander mostly re-assigns current sector — reassignment rate must be measured | R1 §7, R2 §6.2 |
| **M3** | No communication-based MARL baseline (CommNet/TarMAC) despite the paper's thesis being that flat methods lack coordination | R2 §11, R3 §9.1 |
| **M4** | "Steps per fire update = 60" with 150 max steps ⇒ ≤ 2 fire updates per episode — irreconcilable with "dynamically evolving hazard landscape" framing | R1-W7, R2 §10 |
| **M5** | Cell2Fire vs. custom-wrapper ambiguity: "cascade probability" and "steps per fire update" are not FBP/Cell2Fire concepts — clarify what is validated physics vs. reimplemented | R1 §10, R2 §11 |
| **M6** | Evaluation breadth below AAAI bar: one simulator, two 32×32 maps, N = 3; scale study N ∈ {3, 6, 10}, grid ∈ {32², 64²} requested | R3 §6, §9.5 |
| **M7** | Flat-MARL failure asserted but not dissected — behavioral analysis (where agents go, what they treat) would substantiate "coordination collapse" and raise impact | R3 §3 |
| **M8** | Statistical unit ambiguity: Welch tests over 5 seed means or 75 episodes? df not reported | R1-M2 |
| **M9** | Novelty framing inflated: compliance = 1.000 by construction presented as a finding; 16³ = 4,096 joint action space called "large"; contribution reads as heuristic imitation — AC recommends reframing as benchmark + diagnostic study with the hierarchy as reference solution | R2 §3–4, AC-Consensus |
| **M10** | Recency check needed: confirm no closer 2024–2026 prior work on hierarchical dispatch for wildfire suppression before claiming distinctiveness | R3 §2 |

### Low — presentation

| ID | Issue | Where raised |
|----|-------|--------------|
| **L1** | Reward column: bolding differences of ~0.1–1.0 on a ~−550 base implies meaningless precision — decompose the reward scale or drop the column | R1-M1 |
| **L2** | Eq. (7) likely overfull in two-column layout — split or use `multline` | R1-M4 |
| **L3** | §9 text says TRS "> 0.95" but Figure 4 shows Saudi→California = 0.95 exactly (should be ≥) | R1-M5 |
| **L4** | Figure path hygiene — **verified already correct**: `Figures/*.png` exist under `AAAI Template/Figures/`; no action beyond a compile check | R1 §6 |
| **L5** | Limitations §10 should add: single learned baseline, teacher-parity ceiling, reward–metric circularity (evaluation metric is the training reward) | R3 §8 |

---

## 2. Strategy Decision (adopted up front)

The three reviewers diverge on framing; the Area Chair endorses Reviewer #3's reading as the constructive path. This plan **adopts the AC's reframe-plus-rerun strategy**:

> Position the paper as an **infrastructure-aware wildfire-coordination benchmark** with a **diagnostic study of why flat CTDE fails** (properly trained MAPPO + QMIX + one communication baseline), presenting the BC+KL hierarchy as a **strong reference solution** rather than the headline methods contribution.

Consequences baked into the phases below:

1. The benchmark (environment, GIS layers, metrics, release) becomes a first-class contribution → H5, M5, M6 get elevated attention (Phases 5–6).
2. The "coordination collapse" diagnosis must survive fair baselines → Phase 2 is the load-bearing experimental phase.
3. The hierarchy no longer needs to beat the heuristic to justify the paper — but the honest teacher-parity framing must be consistent everywhere (Phase 8).
4. If, after Phase 2, properly trained flat/coordinated baselines *do* solve the task, the diagnosis claim falls; the fallback (documented, per the V2 plan's convention) is a pure benchmark paper with the failure analysis removed. This is a Phase 2 gate outcome, not a failure of the plan.

---

## 3. Phase Dependency Map

```
0 Forensic audit ─► 1 Desk-reject elimination (LaTeX/bib)
        │
        └─► 2 Baseline rehabilitation (MAPPO, QMIX, CommNet/TarMAC)  ◄─ hard experimental gate
                 └─► 3 Ablation rerun under main protocol (+ BC-only)
                          └─► 4 Statistical reanalysis & claims recalibration
                                   └─► 5 Generalization & transfer redesign
                                            └─► 6 Environment & data transparency
                                                     └─► 7 Figures & qualitative evidence
                                                              └─► 8 Rewrite & reframe
                                                                       └─► 9 Certification & submission
```

Phases 1 and 2 can run in parallel (writing vs. compute). Estimated total: **4–6 weeks** (matches the AC's estimate; the environments are small, so the constraint is protocol discipline, not compute).

---

# Phase 0 — Forensic Audit & Ground Truth

**Effort:** 2–3 days · **Issues initiated:** C1, H1, H3, M4, M5 (root-causing) · **No manuscript edits in this phase.**

## Objective

Before anything is rerun or rewritten, establish the provenance of every number in the manuscript. The review's most damaging observation is not any single error but that six independent inconsistencies destroyed reviewer trust (R2 §7). Trust is rebuilt by tracing, not by patching.

## Tasks

1. **Build a provenance table** (`docs/paper_provenance.md`): for every number in Tables 1–6, every figure, and every in-text statistic (p-values, d, latency 1.67, drift 4.32, TRS values), record the producing artifact under `results/` (runs, phase8, phase13, phase15), the config, seed set, episode count, and checkpoint hash. Mark each entry `TRACED` / `UNTRACEABLE` / `PROTOCOL-MISMATCH`.
2. **Root-cause C1:** identify exactly what produced Table 3's Full System row (17.4/0.400) vs. Table 1's (20.28/0.366) — different checkpoint, single seed, different episode count, or an earlier code state. Write the finding down verbatim; it determines whether Phase 3 is a rerun or also a bug hunt.
3. **Root-cause H1:** from training logs/configs, establish how many environment steps flat MAPPO actually saw, which reward it was trained on, and whether the bit-identity with No-Op is policy collapse (all-stay/no-treat actions) or an evaluation-pipeline bug (e.g., wrong checkpoint loaded, actions ignored).
4. **Root-cause H8:** determine why three distinct ablations produced byte-identical results and why the entropy ablation is identical to the full system (single seed? discrete outcome space? copy-paste error in the results table?).
5. **Locate the QMIX runs (H3):** the checkpoints exist per Appendix D — find their training/eval artifacts and assess whether they can simply be evaluated under the main protocol in Phase 2.
6. **Verify simulator facts (M4, M5):** from the code, document precisely (a) how many fire-state updates occur in a 150-step episode, and (b) which components are Cell2Fire proper vs. the custom wrapper (cascade probability, update cadence, treatment model). No spin — just the facts, for use in Phases 5–6.
7. **Record the strategy decision** of §2 above (benchmark + diagnosis reframe) in `Phases.md` as the Phase 0 outcome, so later phases don't relitigate it.

## Success Criteria (gate to Phase 1/2)

- [ ] Provenance table exists covering **100% of reported numbers**; each row is `TRACED`, `UNTRACEABLE`, or `PROTOCOL-MISMATCH` — no blanks.
- [ ] The Table 1 / Table 3 divergence (C1) has a written root cause, reproducible from artifacts.
- [ ] The MAPPO = No-Op bit-identity (H1) has a written root cause distinguishing "policy collapsed to inaction" from "evaluation bug".
- [ ] The ablation identical-values anomaly (H8) has a written root cause.
- [ ] Every `UNTRACEABLE` number is scheduled for rerun in Phase 2 or 3 (mapping recorded).
- [ ] Fire-update cadence and Cell2Fire-vs-wrapper boundary are documented from code, not memory.

---

# Phase 1 — Desk-Reject Elimination (LaTeX, Bibliography, Checklist Skeleton)

**Effort:** 0.5–1 day · **Issues closed:** C4, C5, L2, L3, L4 · **Issues advanced:** C3 (skeleton only — final answers need Phases 2–6 data)

## Objective

Remove every mechanical ground for desk rejection identified in the Final Verdict. This phase is pure manuscript hygiene and can run in parallel with Phase 2 compute.

## Tasks

1. **Preamble compliance (C5)** in `AnonymousSubmission2027.tex`:
   - `\usepackage[submission]{aaai2027}` (restores `\pdfinfo` function and enforces submission-mode formatting).
   - Remove `\usepackage{times}`, `\usepackage{helvet}`, `\usepackage{courier}` — the .sty loads `newtxtext`/`helvet`/`courier` itself and explicitly forbids `times`.
   - Remove manual `\setlength{\pdfpagewidth}{8.5in}` / `\setlength{\pdfpageheight}{11in}` — the style owns page geometry.
   - Audit `\usepackage{caption}`, `subcaption`, and `\setcounter{secnumdepth}{2}` against the AAAI-27 author kit's forbidden-modifications list; remove or replace anything prohibited.
   - Re-verify the full package list against the author kit (e.g., `algorithm`/`algorithmic` allowed forms).
2. **Bibliography (C4):** either add a complete, verified BibTeX entry for `altameem2022wildfire` (locate the actual paper; if it cannot be verified to exist, this is a hallucinated citation) or remove the citation from §2 and substitute a real grid-based wildfire-RL reference. Run the repo's `bibtex-validator` skill/tooling over `aaai2027.bib` to catch any further dead or malformed entries.
3. **Equation width (L2):** reformat Eq. (7) with `multline` or a split `align` so it fits one column; compile and confirm zero overfull `\hbox` warnings on that page.
4. **TRS wording (L3):** change §9 "TRS > 0.95" to "TRS ≥ 0.95" (interim fix; §9 is rewritten in Phase 5 anyway).
5. **Figure paths (L4):** confirmed already correct — verify via clean compile only.
6. **Checklist skeleton (C3, partial):** replace all 34 "Type your response here" fields with either a real answer (where the paper already contains the information) or an explicit `TODO(phase-N)` marker naming the phase that produces the answer. Zero fields may remain in template state.

## Success Criteria (gate to Phase 3 dependency; Phase 2 may already be running)

- [ ] `pdflatex` + `bibtex` complete with **zero errors, zero undefined citations, zero missing references** (`grep` the `.log` for "undefined" and "Warning--" in `.blg`).
- [ ] Preamble contains `[submission]` option; `times`/`helvet`/`courier` and manual page-size settings are gone; a written line-by-line diff against the AAAI-27 author-kit template exists and shows no forbidden modifications.
- [ ] Eq. (7) page compiles with no overfull hbox > 1pt.
- [ ] `ReproducibilityChecklist.tex` contains zero "Type your response here" strings; every field is answered or carries a phase-tagged TODO.
- [ ] All three figures resolve in the compiled PDF.

---

# Phase 2 — Baseline Rehabilitation (the load-bearing experimental phase)

**Effort:** 5–10 days · **Issues closed:** H1, H3, M3 · **Prereq:** Phase 0 root causes.

## Objective

Replace the non-credible baseline story with one that can survive adversarial review. The paper's central diagnostic claim ("flat CTDE fails; strategic coordination is the bottleneck") is currently confounded by an apparently untrained MAPPO. After this phase, every baseline is either demonstrably converged or demonstrably unable to learn the task despite a fair budget — and either outcome is reportable.

## Tasks

1. **Retrain flat MAPPO (H1)** with a defensible budget (order-of-magnitude increase over 10k steps; justify the chosen budget against MARL literature norms for comparable tasks), trained on **the same WEL/ISR-shaped reward** as the commander (Eq. 1) to remove the objective-level confound R2 identified. Log and save training curves per seed.
2. **Train/evaluate QMIX (H3)** under the identical protocol. Checkpoints already exist per Appendix D — Phase 0 determines whether they are usable or must be retrained. Either way, QMIX results enter Table 1, or the hashes are deleted; ghost artifacts are not an option.
3. **Add one communication baseline (M3):** CommNet or TarMAC (pick one; TarMAC if attention infrastructure exists, CommNet otherwise). This is the "natural rebuttal experiment" R3 demanded: if flat methods fail for lack of coordination, a communication channel is the minimal fix to test.
4. **Evaluate all baselines under the main protocol:** 5 seeds × 15 episodes, bootstrap 95% CIs — including No-Op and every learned baseline (fixes the asymmetric-reporting suspicion, R2 §5.3).
5. **Convergence evidence:** for each learned baseline, produce a training-curve figure (mean ± CI across seeds) demonstrating plateau, destined for a new appendix.
6. **Behavioral instrumentation (feeds M7/Phase 7):** log per-step agent positions, treat actions, and distance-to-nearest-asset for all baselines during evaluation, so Phase 7 can dissect *how* flat methods fail rather than merely that they fail.
7. **Action-distribution sanity check:** report each baseline's action histogram; if any policy is ≥ 99% "stay", state it explicitly rather than letting reviewers infer it from metric identity.

## Success Criteria (gate to Phase 3)

- [ ] Every learned baseline has per-seed training curves showing plateau (or documented divergence) at the disclosed budget; budgets and rewards are recorded in the config and will appear in Table 4.
- [ ] All Table 1 rows — including No-Op and all baselines — carry 5-seed × 15-episode means with bootstrap 95% CIs. **No row is a single run.**
- [ ] No baseline result is bit-identical to No-Op *unless* the action logs prove genuine behavioral collapse, in which case that evidence is retained for the paper.
- [ ] QMIX appears in the results with full protocol, **or** its checkpoint hashes are removed from Appendix D (decision recorded).
- [ ] A communication baseline (CommNet or TarMAC) is evaluated under the identical protocol.
- [ ] **Strategy checkpoint:** the "coordination collapse" claim is re-assessed against the new results. Outcome recorded in `Phases.md`: (a) claim survives with fair baselines → proceed as diagnosis + benchmark paper; (b) fair baselines solve the task → drop the diagnosis claim, proceed as benchmark + reference-solutions paper. Either way, Phase 3 proceeds.

---

# Phase 3 — Ablation Rerun Under the Main Protocol

**Effort:** 2–4 days · **Issues closed:** C1, H2, H8 · **Prereq:** Phase 2 (shared eval harness).

## Objective

Make the ablation table (Table 3) an instrument of evidence instead of a source of contradiction. Every row reruns under the exact main protocol, the missing BC-only row is added, and the full-system row must reproduce Table 1 — the single most cited flaw in all three reviews.

## Tasks

1. **Rerun all ablations** (w/o BC, w/o KL, w/o entropy, w/o WEL/ISR reward, w/o target-seeking) at 5 seeds × 15 episodes with bootstrap CIs, from the same checkpointing discipline as the main results.
2. **Add the BC-only row (H2):** Phase-1-only commander (no RL fine-tuning), same protocol. This isolates what KL-regularized REINFORCE actually buys over pure imitation — R2's "single most important missing ablation". Report the Hierarchical-vs-BC-only significance test.
3. **Full-system consistency (C1):** the ablation table's "Full System" row must be *the same experiment* as Table 1's Hierarchical row — same checkpoints, same seeds, same episodes — so the numbers are identical by construction, not approximately equal by luck.
4. **Resolve the identical-values anomaly (H8):** with 5 seeds, byte-identical rows should disappear; if any ablations remain statistically indistinguishable, report that honestly with the test, and explain the mechanism (e.g., KL anchor dominating).
5. **Add per-row significance tests** vs. the full system (Welch's t on seed means, consistent with Phase 4's protocol decisions).

## Success Criteria (gate to Phase 4)

- [ ] Table 3's Full System row is **numerically identical** to Table 1's Hierarchical row (same underlying runs) — the C1 contradiction is structurally impossible, not just currently absent.
- [ ] Every ablation row has 5-seed CIs and a significance test vs. full system.
- [ ] A BC-only row exists; the RL fine-tuning contribution is quantified with an effect size and CI (whether positive, null, or negative).
- [ ] No two distinct ablations share byte-identical results without a written, evidence-backed explanation in the paper.
- [ ] All new numbers are entered into the Phase 0 provenance table as `TRACED`.

---

# Phase 4 — Statistical Reanalysis & Claims Recalibration

**Effort:** 1–2 days (analysis only; no new training) · **Issues closed:** H7, M8, and the analysis halves of C2 and M2 · **Prereq:** Phases 2–3 data.

## Objective

Replace every statistically indefensible inference with a defensible one, using the data now in hand. No prose styling yet — this phase produces the numbers and the claim wording constraints that Phase 8 must honor.

## Tasks

1. **Equivalence testing (H7):** replace "statistical parity via p > 0.05" with TOST (two one-sided tests) or CI-based equivalence bounds for Hierarchical vs. Value-First, per region. Pre-register the equivalence margin (e.g., ±5% WEL) and justify it in text. If equivalence is *not* established on Saudi (plausible given d = −1.23), the claim becomes "comparable but not established equivalent; the heuristic may retain a small edge" — wording locked here.
2. **Statistical unit & df (M8):** decide and document the unit of analysis (5 seed means — the conservative, standard choice), report n and df for every test, and recompute all p-values/effect sizes accordingly. One protocol, applied uniformly to Tables 1, 3, and 6.
3. **Degenerate-variance effect sizes (C2 support):** wherever a baseline has (near-)zero variance, report Cohen's d with an explicit caveat or omit d in favor of the mean difference with CIs. No more d = 13.28 presented as a triumph.
4. **Region-accurate headline numbers (C2):** compute the final per-region Hierarchical-vs-flat-MARL comparisons from Phase 2's rehabilitated baselines. These — not the old Saudi-only numbers — are what the abstract may cite. If California remains non-significant, the abstract must say "in the petroleum-infrastructure region" or report both regions explicitly.
5. **Reassignment rate (M2):** compute from Phase 2's logged trajectories: fraction of macro-steps where an agent's assigned sector differs from its current sector, and fraction where the assignment *changes* from the previous macro-step. This answers R1's Question 2 with data.
6. **Reward decomposition (L1 support):** decompose the ~−550 reward base into its constant and policy-dependent components so Phase 8 can either present a meaningful reward column or drop it with justification.

## Success Criteria (gate to Phase 5)

- [ ] Every statistical claim destined for the paper maps to a test with stated n, df, test type, and (where applicable) pre-registered equivalence margin — assembled in a claims-to-evidence table (extends the Phase 0 provenance doc).
- [ ] TOST/equivalence results exist for both regions; the permissible wording for the "matches the expert" claim is written down and frozen.
- [ ] No effect size on a (near-)degenerate distribution is reported without its caveat.
- [ ] Reassignment-rate statistics are computed for both regions.
- [ ] The abstract-eligible claim set (what may be asserted, with which numbers) is explicitly listed for Phase 8.

---

# Phase 5 — Generalization & Transfer Redesign

**Effort:** 3–5 days · **Issues closed:** M1, M6 (core), advances M4 framing · **Prereq:** Phase 4 protocol decisions.

## Objective

Replace the unfalsifiable TRS metric with generalization evidence that can actually fail, and widen the evaluation axis enough to clear the "toy scale, one setting" objection.

## Tasks

1. **Retire or fix TRS (M1).** Preferred: replace §9 with held-out-scenario generalization — evaluate trained policies on (a) held-out ignition points, (b) shifted wind regimes, and (c) unseen asset layouts, within region and cross-region. If TRS is retained anywhere, renormalize it against the No-Op baseline (regret-style: policy improvement over No-Op, transferred vs. native) so a failed transfer scores near 0, not 0.95.
2. **Falsifiability check:** compute the new generalization metric for No-Op. It must score demonstrably poorly — this is the direct answer to "a metric that cannot fail" and goes in the paper as a sanity row.
3. **Scale study (M6):** N ∈ {3, 6, 10} agents and grid ∈ {32², 64²} for the hierarchy, the heuristic, and the strongest rehabilitated baseline. Even partial results (e.g., 64² with N = 6 only) materially answer R3 §6; document any configuration that is computationally out of reach and say so in Limitations.
4. **Fire dynamics honesty (M4):** using Phase 0's cadence facts, either (a) increase fire-update frequency so the hazard genuinely evolves within an episode (then rerun headline configs — check impact before committing), or (b) keep the cadence and rewrite all "dynamically evolving hazard" framing to match reality (e.g., "quasi-static fire fronts with episodic advance"). Decision is experimental first: measure how results change under faster fire updates on one region before choosing.

## Success Criteria (gate to Phase 6)

- [ ] The generalization section's metric assigns a clearly bad score to No-Op (falsifiability demonstrated in-paper).
- [ ] Held-out ignition/wind/layout results exist for both regions with the standard protocol (seeds, CIs).
- [ ] Scale-study results exist for at least two agent counts and two grid sizes, or a documented infeasibility note for the missing cells.
- [ ] The fire-update-cadence decision (change dynamics vs. change framing) is made from measured evidence and recorded; the paper's hazard-dynamics language now matches the simulator's actual behavior.
- [ ] Figure 4 (transfer heatmap) is regenerated or retired accordingly.

---

# Phase 6 — Environment & Data Transparency

**Effort:** 1–2 days · **Issues closed:** H5, H6, M5 · **Prereq:** Phase 5 (final environment configuration frozen).

## Objective

Make the benchmark contribution — now the paper's headline per the §2 strategy — actually documentable, releasable, and reproducible. This is where "not reproducible from the manuscript" (R1 §9) gets reversed.

## Tasks

1. **GIS provenance section (H5):** for each region, document the data source (dataset name, provider, access date), native resolution, the exact preprocessing pipeline down to the 32×32 grid (projection, aggregation, asset-value assignment), licensing, and the release plan. If the asset values are stylized rather than measured, say so explicitly and describe the stylization rules — reviewers punish vagueness harder than honest simplification.
2. **Simulator boundary appendix (M5):** rewrite Appendix E from Phase 0's code-level findings: state precisely which components are Cell2Fire/FBP (fuel models, spread mechanics) and which are the custom wrapper (cascade probability, update cadence, treatment/suppression model, PettingZoo layer). Retitle claims accordingly — "Cell2Fire-based" not "validated Cell2Fire physics" if the wrapper modifies dynamics.
3. **Compute disclosure appendix (H6):** hardware (CPU/GPU, RAM), OS, wall-clock per training run and per evaluation sweep, library versions (torch, gymnasium, pettingzoo, Cell2Fire commit), and the **actual seed values** for training and evaluation.
4. **Release package:** assemble the code + configs + checkpoints + GIS layers (or their generation scripts) + manifest with SHA-256 hashes; regenerate Appendix D's hash table from the final checkpoints so it matches the release exactly (and now includes every policy reported, only policies reported).
5. **Finish the reproducibility checklist (C3):** replace every `TODO(phase-N)` marker with the real answer, now that all supporting content exists in the paper.

## Success Criteria (gate to Phase 7)

- [ ] The GIS section answers, in-paper: source, resolution, preprocessing, license, release — all five, per region.
- [ ] Appendix E draws an unambiguous line between validated physics and custom wrapper; no FBP-foreign concept is attributed to Cell2Fire.
- [ ] Compute appendix complete, including literal seed values.
- [ ] `ReproducibilityChecklist.tex` contains zero TODOs and zero template strings; every answer is consistent with the paper's content (spot-check 100% of "yes" answers against the section they cite).
- [ ] Appendix D hashes regenerated from the release package; set of hashed checkpoints == set of reported policies.
- [ ] A clean-environment smoke test (fresh venv, released code, one seed) reproduces one headline number within tolerance.

---

# Phase 7 — Figures & Qualitative Evidence

**Effort:** 2–3 days · **Issues closed:** H4, M2 (figure half), M7 · **Prereq:** Phases 2–5 (final policies and logs).

## Objective

Every figure must support its own caption under adversarial reading. Two of three current figures actively contradict theirs (R1 §6) — worse than no figure.

## Tasks

1. **Regenerate Figure 2 (H4):** left panel must be an actual rehabilitated Flat-MARL rollout (not No-Op); choose a timestep where the footer metrics *differ* between panels; annotate the episode step and explain the WEL scale relative to Table 1 (early-episode snapshot) in the caption.
2. **Regenerate Figure 3 (M2):** select a real dispatch event where the commander routes at least one agent across sectors; overlay the Phase 4 reassignment-rate statistic in the caption so the "distributes agents to distinct high-value sectors" claim is backed by both the image and a number. If the honest finding is that reassignment is rare after initial placement, show initial placement and *say that* — it remains a valid strategic behavior.
3. **New figure — flat-MARL failure analysis (M7):** from Phase 2's behavioral logs, visualize where flat agents spend time vs. asset locations (occupancy heatmap or distance-to-asset distributions) contrasted with the hierarchy. This converts "coordination collapse" from assertion to evidence and directly addresses R3's impact critique.
4. **New figure — baseline training curves** (from Phase 2, appendix).
5. **Regenerate/retire Figure 4** per Phase 5's outcome.
6. **Caption–content audit:** a fresh reader (or a review pass with the `peer-review` skill) checks every figure against its caption and against the body text that cites it; every claimed visual feature must be visibly present.

## Success Criteria (gate to Phase 8)

- [ ] Zero caption–content mismatches across all figures (audited independently, findings logged).
- [ ] Figure 2 panels show different policies with visibly/numerically different outcomes, timestep stated.
- [ ] Figure 3 depicts a genuine cross-sector dispatch or is honestly recaptioned around initial placement; reassignment rate appears in caption or §7 text.
- [ ] The flat-MARL failure-analysis figure exists and is cited by the Finding-1 discussion.
- [ ] All figures regenerated at publication resolution from checked-in scripts (paths under `Figures/`, scripts under `scripts/` or `src/wildfire_marl/viz/`).

---

# Phase 8 — Manuscript Rewrite & Reframe

**Effort:** 3–5 days · **Issues closed:** C2 (text), M9, M10, L1, L5, plus all wording locked in Phase 4 · **Prereq:** all experimental phases complete.

## Objective

Rewrite the paper around the §2 strategy with every claim bounded by Phase 4's claims-to-evidence table. The review's writing-quality score was the paper's best (7/10) — the problem is claim calibration, not prose.

## Tasks

1. **Abstract & Introduction (C2, M9):** reframe contributions in the AC's ordering — (i) infrastructure-aware wildfire-coordination benchmark on documented GIS data with validated-physics backend, (ii) diagnostic study of flat CTDE failure with fair baselines (as supported by Phase 2's outcome), (iii) BC+KL hierarchical reference solution with compliance-by-design, (iv) rigorous statistical protocol. Every abstract number comes from the Phase 4 abstract-eligible list; region qualifiers mandatory where results are region-specific.
2. **Kill inflated framing (M9):** 16³ action space no longer described as "large/combinatorial" (motivate BC warm-start via sparse delayed reward instead — which the ablation genuinely supports); compliance = 1.000 presented as a design guarantee verified by diagnostics, not a discovered finding; "compliance gap" positioned as a design principle with measurement framework, not a conceptual discovery.
3. **Literature recency pass (M10):** search 2024–2026 literature for hierarchical dispatch / commander-style MARL for wildfire suppression; add and position any close prior work in §2 *before* claiming distinctiveness. Also give operational-research suppression planning more than one dismissive sentence (R3 §5b) — one honest paragraph on why sequential decision-making under partial observability motivates RL relative to MIP planning.
4. **Results & findings rewrite:** Finding 1 rewritten around rehabilitated baselines + behavioral-analysis figure; Finding 2 rewritten in TOST language; Finding 3 region-accurate with both regions' numbers stated; new finding for the BC-only comparison (what RL fine-tuning buys — report honestly whichever direction it goes).
5. **Table 1 presentation (L1):** decomposed reward or dropped column per Phase 4; bolding only where differences are statistically supported.
6. **Limitations (L5):** add single-learned-baseline history (now mitigated), teacher-parity ceiling, reward–metric circularity, fire-update cadence (if kept quasi-static), and scale boundaries from Phase 5.
7. **Consistency sweep:** regenerate every in-text number from the final results artifacts (ideally via the existing `build_report_tables` provenance tooling) so text, tables, and figures cannot drift.

## Success Criteria (gate to Phase 9)

- [ ] Every quantitative claim in abstract, findings, and conclusion appears in the Phase 4 claims-to-evidence table with a passing test behind it — verified line by line.
- [ ] The word "dramatically" (and equivalents) appears only where both regions support it, or is qualified by region.
- [ ] No claim of equivalence/parity without a TOST result; no effect size without variance context.
- [ ] Contributions list matches the benchmark + diagnosis + reference-solution framing; no contribution restates a by-construction property as a finding.
- [ ] §2 contains the 2024–2026 recency check outcome and the expanded OR-planning positioning.
- [ ] A full-document grep-level consistency pass finds zero contradictions between any two tables/figures/paragraphs (rerun the R1 §6 consistency audit checklist against the new draft — all six original inconsistencies must be structurally resolved).

---

# Phase 9 — Certification, Internal Re-Review & Submission

**Effort:** 2–3 days · **Issues closed:** final C3 verification; end-to-end quality gate.

## Objective

Simulate the review that just rejected the paper, against the revised paper, before anyone external sees it — then package for the chosen venue.

## Tasks

1. **Internal adversarial re-review:** run the same full-review protocol that produced `Peer_Review_Report.md` (e.g., via the `peer-review` skill) on the revised manuscript. Every Critical/High finding from the original report must be verifiably closed; any *new* Critical/High finding blocks submission.
2. **Reproducibility certification:** clean-machine end-to-end run (per Phase 6's package) reproducing Table 1's Hierarchical rows and at least one baseline row from released artifacts; record in `Phases.md` per the Phase-15 convention.
3. **Checklist cross-validation (C3 final):** every checklist "yes" is traced to the specific paper section/appendix that substantiates it; no aspirational answers.
4. **Venue decision:** with final results in hand, choose per the AC's guidance — AAAI 2028 main track or AAMAS main track if the diagnosis claim survived Phase 2 strongly; AAAI AI-for-Social-Impact track if the benchmark carries the paper. Adjust formatting/anonymization to the chosen venue's kit.
5. **Final desk-check:** page limits, anonymization (no repo links that deanonymize; supplementary packaged separately), figure resolutions, PDF metadata, checklist attached.

## Success Criteria (submission gate)

- [ ] Internal re-review yields **zero Critical and zero High findings**; Medium findings are either fixed or consciously accepted with a written rationale.
- [ ] Every row of the Section 1 issue register above is marked closed with a pointer to the fixing artifact/commit/section — the register is the exit checklist.
- [ ] Clean-machine reproduction succeeds within stated tolerance.
- [ ] Checklist, compute disclosure, provenance, and release package are mutually consistent.
- [ ] Venue chosen, kit compliance verified, submission package assembled.

---

## 4. Traceability Matrix (issue → phase)

| Issue | Phase(s) | Issue | Phase(s) |
|-------|----------|-------|----------|
| C1 | 0, 3 | M1 | 5 |
| C2 | 4, 8 | M2 | 4, 7 |
| C3 | 1, 6, 9 | M3 | 2 |
| C4 | 1 | M4 | 0, 5 |
| C5 | 1 | M5 | 0, 6 |
| H1 | 0, 2 | M6 | 5 |
| H2 | 3 | M7 | 2, 7 |
| H3 | 0, 2 | M8 | 4 |
| H4 | 7 | M9 | 8 |
| H5 | 6 | M10 | 8 |
| H6 | 6 | L1 | 4, 8 |
| H7 | 4 | L2 | 1 |
| H8 | 0, 3 | L3 | 1 |
|  |  | L4 | 1 (verified) |
|  |  | L5 | 8 |

## 5. Risk Register

| Risk | Phase | Mitigation |
|------|-------|-----------|
| Fairly trained baselines solve the task → "coordination collapse" claim falls | 2 | Pre-committed fallback: benchmark + reference-solutions framing (§2, item 4); the paper survives either outcome |
| TOST shows heuristic significantly better than hierarchy on Saudi | 4 | Honest reporting + BC-only row contextualizes; teacher-parity ceiling already in Limitations; benchmark framing does not require winning |
| BC-only ≈ fine-tuned (RL adds nothing) | 3 | Report as a finding about KL-anchored fine-tuning; strengthens the diagnostic-paper framing; consider a λ-sweep if time allows |
| GIS data cannot be documented to release standard | 6 | Recreate layers from documented public sources, or relabel as "stylized layouts informed by public GIS data" — vagueness is the only unacceptable option |
| Faster fire updates change headline results | 5 | Measured pilot before committing; if results change materially, prefer the honest-framing option (b) for this cycle and note dynamics as future work |
| 64² / N=10 scale runs too slow (Cell2Fire subprocess I/O) | 5 | Partial matrix + documented infeasibility is acceptable per gate; profile first (per V2 plan's known risk) |

---

*Execution log entries for each phase go to `Phases.md`. Nothing in this plan is committed to git by the executor — commits are made by the author.*
