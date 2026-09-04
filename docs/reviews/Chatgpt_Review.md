# AAAI Readiness Roadmap — Wildfire RL Project

# PROJECT OBJECTIVE

Develop a scientifically defensible, reproducible, and AAAI-competitive research framework for:

* cross-regional wildfire reinforcement learning,
* controllability analysis,
* transfer-learning evaluation,
* and multi-agent wildfire suppression.

The roadmap below defines:

* exact development phases,
* required experiments,
* required outputs,
* validation conditions,
* and hard success criteria.

NO PHASE should be considered complete unless all success criteria are satisfied.

---

# PHASE 0 — REPOSITORY & REPRODUCIBILITY FOUNDATION

# Goal

Create a professional research infrastructure suitable for:

* reproducibility,
* experimentation,
* scaling,
* and AAAI artifact expectations.

# Required Tasks

## Repository Engineering

* [x] canonical `src/` package
* [x] Hydra/OmegaConf configs
* [x] CLI pipelines
* [x] experiment tracking
* [x] deterministic seeding
* [x] reusable evaluation modules

## Reproducibility

* [x] pinned dependencies
* [x] environment.yml
* [x] requirements.txt
* [x] Dockerfile
* [x] reproducibility manifests
* [x] GitHub Actions
* [x] automated tests

## Git Hygiene

* [x] strict `.gitignore`
* [x] dataset exclusion
* [x] checkpoint exclusion
* [x] Git LFS strategy

## Documentation

* [x] README
* [x] architecture docs
* [x] reproducibility docs
* [x] data card
* [x] model card

---

# Required Outputs

* [x] clean GitHub repository
* [x] passing tests
* [x] successful CI pipeline
* [x] reproducible experiment scripts
* [x] deterministic seed runs

---

# SUCCESS CRITERIA

ALL must be true:

✅ Repository installs cleanly on fresh machine
✅ Tests pass automatically
✅ `make train/eval/figures` works
✅ Experiments reproducible across seeds
✅ No hardcoded paths
✅ Repo safe to push publicly
✅ No large-file Git failures
✅ Hydra configs fully functional
✅ All notebook logic migrated to canonical modules

---

# PHASE 1 — ENVIRONMENT VALIDATION & CONTROLLABILITY ANALYSIS

# Goal

Determine whether the wildfire environment is meaningfully controllable.

This is the MOST IMPORTANT scientific phase.

---

# Required Tasks

## Environment Audit

Measure:

* [x] suppression coverage ratio
* [x] spread aggressiveness
* [x] ignition dynamics
* [x] stochasticity dominance
* [x] action influence

## Controllability Experiments

Evaluate:

* [x] noop
* [x] random
* [x] nearest-fire heuristic

Measure:

* [x] burned cells
* [x] fire intensity
* [x] containment efficiency
* [x] spread rate

## Action Influence Analysis

Quantify:

* [x] delta fire spread WITH action
* [x] delta fire spread WITHOUT action

## Environment Improvements

Implement:

* [x] stronger suppression radius
* [x] directional wind
* [x] improved spread kernel
* [x] reward redesign

---

# Required Outputs

## Figures

* [x] suppression influence maps
* [x] fire propagation trajectories
* [x] action impact visualizations
* [x] controllability plots

## Tables

* [x] suppression coverage statistics
* [x] baseline comparisons
* [x] controllability metrics

---

# SUCCESS CRITERIA

ALL must be true:

✅ Actions measurably affect fire evolution
✅ PPO influence > noop influence
✅ Environment not dominated entirely by randomness
✅ Fire spread responds to suppression actions
✅ PPO learns non-trivial behavior
✅ Action heatmaps differ from random
✅ Environment visually demonstrates controllability
✅ Reward function produces stable learning signal

---

# PHASE 2 — BASELINE SUITE DEVELOPMENT

# Goal

Establish scientifically credible comparison baselines.

---

# Required Baselines

## Mandatory

* noop
* random
* nearest-fire heuristic
* frontier suppression heuristic
* PPO

## Optional

* greedy suppression
* downwind suppression

---

# Required Tasks

## Baseline Evaluation

Evaluate ALL baselines on:

* Saudi
* California

using:

* identical episode budgets
* identical seeds
* identical evaluation conditions

## Statistical Evaluation

Compute:

* mean
* std
* 95% CI
* Welch t-tests
* Cohen’s d

---

# Required Outputs

## Tables

* baseline comparison tables
* effect-size tables
* statistical significance tables

## Figures

* baseline reward plots
* suppression efficiency plots
* fire containment comparisons

---

# SUCCESS CRITERIA

ALL must be true:

✅ PPO > noop
✅ PPO > random
✅ PPO competitive with heuristics
✅ Differences statistically significant
✅ PPO shows consistent behavior across seeds
✅ Heuristic baselines implemented correctly
✅ Variance remains stable across runs

---

# PHASE 3 — PPO STABILIZATION & POLICY ANALYSIS

# Goal

Ensure PPO policies genuinely learn meaningful wildfire behavior.

---

# Required Tasks

## PPO Improvements

Tune:

* entropy coefficient
* learning rate
* rollout horizon
* reward scaling
* observation normalization

## Policy Diagnostics

Measure:

* entropy curves
* convergence
* action diversity
* spatial coverage
* policy collapse risk

## Visualization

Generate:

* action heatmaps
* suppression trajectories
* temporal fire evolution
* PPO vs heuristic overlays

---

# Required Outputs

## Figures

* entropy plots
* training curves
* action distribution maps
* trajectory visualizations

## Diagnostics

* policy diversity metrics
* action entropy statistics
* convergence analysis

---

# SUCCESS CRITERIA

ALL must be true:

✅ PPO no longer collapses to noop-like behavior
✅ Action entropy remains healthy during training
✅ PPO learns spatially meaningful suppression patterns
✅ PPO trajectories differ from random trajectories
✅ PPO consistently converges across seeds
✅ Training stable under repeated runs

---

# PHASE 4 — TRANSFER LEARNING EVALUATION

# Goal

Scientifically evaluate cross-regional generalization.

---

# Required Experiments

## Transfer Matrix

| Train      | Test       |
| ---------- | ---------- |
| Saudi      | Saudi      |
| Saudi      | California |
| California | California |
| California | Saudi      |

## Required Metrics

* normalized reward
* containment efficiency
* burned cells
* fire intensity
* spread rate

---

# Required Tasks

## Domain Shift Analysis

Quantify:

* fuel differences
* wind differences
* terrain differences
* humidity differences

## Transfer Evaluation

Compute:

* degradation %
* effect sizes
* confidence intervals

---

# Required Outputs

## Figures

* transfer heatmaps
* degradation bar charts
* cross-region policy comparisons

## Tables

* full transfer matrix
* transfer degradation statistics
* significance testing tables

---

# SUCCESS CRITERIA

ALL must be true:

✅ Transfer metrics use normalized comparisons
✅ Transfer degradation measurable and interpretable
✅ Native policies outperform transferred policies
✅ Transfer results statistically robust
✅ Domain shift quantified explicitly
✅ Results reproducible across seeds
✅ Transfer claims scientifically defensible

---

# PHASE 5 — RANDOMIZED GENERALIZATION

# Goal

Prevent memorization and demonstrate robustness.

---

# Required Tasks

## Randomization

Randomize:

* ignition location
* wind seeds
* humidity patterns
* fire spread seeds

## Evaluation

Test on:

* unseen ignitions
* unseen stochastic trajectories
* unseen environmental seeds

---

# Required Outputs

## Figures

* randomized ignition performance
* robustness curves

## Tables

* generalization metrics
* variance analysis

---

# SUCCESS CRITERIA

ALL must be true:

✅ Policies generalize beyond fixed ignition patterns
✅ PPO robust to unseen stochastic trajectories
✅ Performance does not collapse under randomization
✅ Memorization risks substantially reduced

---

# PHASE 6 — MULTI-AGENT SCALING (MARL)

# Goal

Determine whether multi-agent coordination improves controllability.

---

# Required Experiments

## Team Sizes

* 1 agent
* 3 agents
* 5 agents
* 10 agents

under matched training budgets.

---

# Required Tasks

## Coordination Analysis

Measure:

* suppression overlap
* spatial coverage
* coordination efficiency
* specialization emergence

## MARL Transfer

Evaluate:

* Saudi MARL → California
* California MARL → Saudi

---

# Required Outputs

## Figures

* scaling curves
* coordination heatmaps
* coverage maps

## Tables

* MARL scaling performance
* coordination metrics
* transfer degradation

---

# SUCCESS CRITERIA

ALL must be true:

✅ MARL significantly improves containment
✅ Increased coverage improves controllability
✅ Coordination patterns emerge visually
✅ MARL outperforms single-agent PPO
✅ Transfer becomes more meaningful under MARL
✅ Results statistically significant

---

# PHASE 7 — SCIENTIFIC VALIDATION & ABLATIONS

# Goal

Validate that conclusions are scientifically justified.

---

# Required Ablations

## Environment Ablations

* no wind
* no terrain
* no suppression
* low fuel
* high fuel

## PPO Ablations

* entropy off
* normalization off
* reward scaling off

---

# Required Tasks

## Failure Analysis

Document:

* policy failures
* transfer failures
* collapse cases

## Sensitivity Analysis

Evaluate:

* hyperparameter sensitivity
* environment sensitivity
* stochastic sensitivity

---

# Required Outputs

## Tables

* ablation results
* sensitivity results

## Figures

* ablation impact plots
* robustness visualizations

---

# SUCCESS CRITERIA

ALL must be true:

✅ Ablations support scientific conclusions
✅ Results robust to moderate hyperparameter changes
✅ Failure cases documented honestly
✅ Claims supported by evidence
✅ Reviewer-risk issues minimized

---

# PHASE 8 — PAPER WRITING & AAAI POSITIONING

# Goal

Transform experiments into a scientifically defensible AAAI paper.

---

# Required Tasks

## Paper Positioning

Frame the work as:

✅ controllability analysis
✅ cross-regional transfer
✅ multi-agent coordination
✅ geospatial RL benchmark framework

NOT:
❌ perfect wildfire suppression
❌ physically exact wildfire simulation

---

# Required Sections

## MUST INCLUDE

* limitations
* failure analysis
* baseline discussion
* controllability discussion
* transfer degradation analysis
* MARL motivation

---

# Required Figures

## Mandatory

1. Transfer matrix heatmap
2. Baseline comparison
3. Action heatmaps
4. Fire trajectories
5. MARL scaling curves
6. Generalization plots
7. Ablation figures

---

# SUCCESS CRITERIA

ALL must be true:

✅ Claims fully supported by evidence
✅ Figures publication-quality
✅ Statistical analysis rigorous
✅ Limitations honestly discussed
✅ Transfer claims scientifically valid
✅ Reviewer criticisms proactively addressed
✅ Narrative internally consistent

---

# PHASE 9 — AAAI PRE-SUBMISSION AUDIT

# Goal

Stress-test the paper before submission.

---

# Required Tasks

## Simulate Reviewers

Generate:

* harsh AAAI reviews
* reproducibility reviews
* novelty criticism
* methodology criticism

## Artifact Validation

Verify:

* reproducibility
* installation
* figure generation
* experiment reruns

---

# Required Outputs

## Reports

* reviewer simulation report
* artifact audit report
* reproducibility checklist

---

# SUCCESS CRITERIA

ALL must be true:

✅ No critical scientific contradictions remain
✅ Experiments reproducible independently
✅ Reviewer criticisms addressed
✅ Artifact reproducibility verified
✅ Statistical claims defensible
✅ Transfer conclusions robust

---

# FINAL AAAI READINESS CHECKLIST

The project is AAAI-ready ONLY IF:

✅ PPO clearly outperforms baselines
✅ Transfer degradation scientifically measurable
✅ Environment controllability demonstrated
✅ MARL scaling validated
✅ Policies visually meaningful
✅ Results reproducible
✅ Statistical analysis rigorous
✅ Claims modest and defensible
✅ Failure cases documented honestly
✅ Reviewer attack simulation passed
✅ Artifact audit passed
✅ Figures publication-quality
✅ Transfer matrix complete
✅ Generalization tested
✅ Baselines scientifically credible

---

# FINAL EXPECTED PROJECT POSITIONING

The strongest scientifically defensible framing is:

## "Controllability, coordination, and cross-regional transfer in geospatial wildfire reinforcement learning environments."

NOT:

❌ "Perfect wildfire suppression AI"

❌ "Highly realistic wildfire forecasting"

The project’s strongest contribution is:

* controllability analysis,
* transfer evaluation,
* and multi-agent coordination behavior in geospatial RL systems.
