# Phase 4 — Claims-to-Evidence Table (FROZEN)

This document locks what the revised paper may claim and with which numbers. Phase 8's
rewrite may not exceed this list. Statistical protocol everywhere: **unit = seed means,
n = 5 per policy; Welch's t with Welch–Satterthwaite df; pooled-SD Cohen's d (suppressed or
flagged when a distribution is degenerate); bootstrap 95% CIs; TOST margin pre-registered
at ±5% of the reference policy's mean WEL.** Source artifacts:
`results/phase4_peer/{stats_tables,tost_results,reward_decomposition,compliance_reassignment_*}.json`,
`results/phase2_peer/`, `results/phase3_peer/`.

## PERMITTED CLAIMS (abstract-eligible)

| # | Claim (maximum permissible strength) | Evidence |
|---|---|---|
| P1 | The two GIS-derived regions **discriminate sharply between coordination strategies**: concentrated petroleum assets reward communication/strategic dispatch (CommNet WEL 11.43 vs No-Op 27.00), scattered WUI assets reward broad reactive coverage (Local Reactive 15.13 vs No-Op 25.25) — the same policy classes reverse rank across regions | Phase 2 §3 tables; stats_tables.json |
| P2 | **Independent, communication-free MAPPO fails on concentrated infrastructure**: WEL identical to No-Op in all 75 evaluation episodes despite a 100k-step budget and an evaluation-matched training objective; training curve plateaus at failure; behavior logs show 5.0% treat actions executed ~6.8 cells from the nearest asset | train_curves.png; behavior_saudi.json; stats_tables.json |
| P3 | **CommNet outperforms both the expert heuristic and the hierarchy in both regions**: Saudi WEL 11.43 vs Hierarchical 20.28 (diff −8.85, CI [−11.3, −6.1], t=−5.89, df=4.4, p=0.0031, d=−3.72); California 19.17 vs 24.53 (p=0.013, d=−2.09) | stats_tables.json |
| P4 | The hierarchy performs **comparably to its Value-First teacher** — Saudi diff +0.80 WEL (t=1.77, df≈8, p=0.115, d=1.12), California diff −0.08 (p=0.95, d=−0.04) — but **equivalence at ±5% is NOT established** (TOST p=0.355 Saudi, p=0.200 California; n=5 is underpowered for equivalence, and Saudi shows a nominal heuristic edge) | tost_results.json |
| P5 | **RL fine-tuning adds nothing over behavior cloning**: BC-only = Full System (ΔWEL 0.00, p=1.00, d=0.00), TOST equivalence ESTABLISHED at ±5% (p=0.044); mechanism: fine-tuning moves commander weights ≤5×10⁻³ and changes 0/200 greedy decisions vs the retrained control | tost_results.json; phase3 check_identity.py |
| P6 | **BC pretraining and deterministic target-seeking are the load-bearing components**: removing either collapses the system to No-Op level (27.00, p=3.0×10⁻⁵, d=13.3 vs full system — d reported with the degenerate-variance caveat), even when the tactical replacement is the fairly-trained 100k-step MAPPO actor | phase3 ablation_summary_saudi.json |
| P7 | **Compliance is a design guarantee, verified**: rate 1.000 ± 0.000 (both regions); dispatch latency 0.98 ± 0.29 (Saudi) / 1.21 ± 0.62 (California) steps; sector drift 1.99 ± 0.35 / 2.47 ± 1.06 cells — 5 seeds × 15 episodes, archived (supersedes the old unarchived Table 2 values 4.32/4.18 and 1.67/1.72, which were not reproducible) | compliance_reassignment_*.json |
| P8 | **Reassignment honesty** (answers R1 Q2): the commander dispatches an agent to a sector other than its current one in 27.1% (Saudi) / 36.6% (California) of dispatch events; consecutive-dispatch assignment changes: 19.2% / 23.6%. Strategic dispatch = initial placement + periodic reallocation, and must be described as such | compliance_reassignment_*.json |
| P9 | The `total_reward` column is dominated by a policy-independent burn constant: policy effects are 0.03–6.4% of the baseline magnitude (−549.1 Saudi / −423.1 California). The paper reports Δreward vs No-Op with CIs, or drops the column | reward_decomposition.json |

## RETIRED CLAIMS (must not appear in any form)

| Claim | Reason |
|---|---|
| "Dramatically outperforms flat MARL" | False in both directions after rehabilitation: fair MAPPO improves on California; CommNet beats the hierarchy everywhere |
| "Statistically matches the expert" / "statistical parity" | TOST failed in both regions; only "comparable, equivalence not established" is permitted (P4) |
| "Coordination collapse" as a general property of flat MARL | Survives only as the narrowed P2 (communication-free MAPPO, concentrated assets) |
| "KL regularization prevents catastrophic forgetting" (+4.6%) | The RL stage is inert (P5); there is nothing to forget |
| Compliance gap "eliminated" as a *finding* | By construction; P7 wording (design guarantee, verified) only |
| "TRS > 0.95" cross-region transfer | Metric vacuous + source run broken (Phase 0 H9); §9 is replaced in Phase 5 |
| Cohen's d vs zero-variance baselines without caveat | d=13.3-style values only with the explicit degenerate-variance note (P6) |
| Old Table 2 values (4.32/4.18 drift, 1.67/1.72 latency) | Not reproducible; superseded by P7's archived values |
| Old Table 4 hyperparameters (BC 500/50, RL 200, λ=0.1, β=0.01, lr 3e-4) | Contradicted by shipped checkpoints; report embedded configs (Phase 6) |

## Statistical-protocol statements for the paper (M8, frozen)

- "All results are means over 5 evaluation seeds (15 episodes each; the seed mean is the
  statistical unit, n = 5). Uncertainty: bootstrap 95% CIs. Tests: Welch's t with
  Welch–Satterthwaite df (reported per test). Effect sizes: pooled-SD Cohen's d, omitted
  where a distribution is degenerate. Equivalence: TOST with a pre-registered ±5% margin."
- Sign convention: differences are policy − reference; lower WEL is better.
