# Forensic Code-Level Audit — Wildfire RL Project (Round 2)

> Adversarial review: written to *break* the experiments. Every claim below is grounded in
> the actual notebook source (file + cell). Read together with `REPOSITORY_AUDIT_AND_PLAN.md`
> (which covers the repository-engineering redesign, Phases 3–8). This document concentrates on
> **experimental integrity, reproducibility, and the validity of the scientific claims** — the
> issues that determine whether this work survives peer review and artifact evaluation.

---

## A. The reward function and environment dynamics (read from source)

All six environment classes (`SaudiWildfireEnv`, `CaliforniaWildfireEnv`, `SaudiEvaluationEnv`,
`CaliforniaEvaluationEnv`, plus the MARL/ablation variants) are **logic-identical** — same
constants, same reward, same dynamics — differing only in the hardcoded tensor path. So the
report's "identical architecture / identical dynamics across regions" claim is **true**, but only
because the code was copy-pasted (no shared module → future drift is inevitable).

Core dynamics (`05_train_ppo_saudi_100k`, cell 5; identical in `06`, `07`, `10`):

```python
spread_prob = 0.01 + 0.15*fuel + 0.08*wind_factor + 0.08*terrain_factor
wind_factor = (state[2, ni, nj] + state[3, ni, nj]) / 2     # wind_x + wind_y at the TARGET cell
if np.random.rand() < spread_prob:                          # global RNG, NOT self.np_random
    new_fire[ni, nj] = min(1.0, new_fire[ni, nj] + 0.12)
...
reward = -float(total_fire) + suppression_bonus            # total_fire = np.sum(state[0])
```

### A1 — "Wind-aware spread" is effectively a no-op for direction 🔴
`wind_factor` is `(wind_x + wind_y)/2` evaluated at the **target** cell and added **identically to
all four neighbours**. It has no relationship to the direction of propagation. Wind therefore
cannot bias spread north/south/east/west — it is an isotropic scalar bump.
- **Why it matters:** the report lists "wind-aware spread" and "Shamal wind influence" as a
  feature and a Saudi-specific dynamic. The code does not implement directional wind.
- **Consequence:** a reviewer who reads the env will flag the claim as unsupported; any
  conclusion attributing regional differences to wind is unfounded.
- **Fix:** compute a directional dot-product between the wind vector and the spread offset
  `(ni-i, nj-j)`; document the spread model with an equation.

### A2 — Reward is unnormalised total fire 🔴 (this is the crux of the transfer claim)
`reward = -Σ fire`, summed over up to 200 steps. Episode reward scales with the *fire load*,
which is dominated by **fuel density** (`0.15*fuel` is the largest spread term). California's NDVI
fuel is far denser than Saudi's desert fuel, so California episodes carry ~2× the fire and ~2× the
(negative) reward **independent of any policy**.
- **Consequence:** the absolute reward gap Saudi (−40k) vs California (−86k) measures *fuel
  density*, not *policy quality* and not *transfer*. See §C — this single design choice invalidates
  the headline "cross-regional generalization" result as currently framed.
- **Fix:** report a policy-relative metric (e.g. *fraction of initial fire contained*, or reward
  normalised by a no-op baseline on the same scenario); never compare raw reward across
  environments with different fire loads.

### A3 — Agent is near-powerless; no evidence it beats doing nothing 🔴
The agent suppresses a 3×3 patch (`state[0, nx, ny] *= 0.2`) and moves one cell per step, while
fire spreads across the whole 32×32 grid stochastically every step. With ~500–800 burning cells,
a single 9-cell suppression per step is negligible.
- **Consequence:** it is entirely plausible the learned policy ≈ a no-op. **Nothing in the repo
  rules this out** (see §B3 — there is no baseline control).
- **Fix:** add random-action and no-op baselines on identical scenarios; the *delta* vs baseline is
  the only meaningful performance signal.

### A4 — CNN has no spatial downsampling → 200 MB checkpoints 🟠
`Conv(7→32,3,pad1) → Conv(32→64,3,pad1) → Flatten → Linear(64*32*32=65536, 256)`. No pooling, no
stride. The first linear layer alone is ~16.7M params; with PPO's Adam state this produces the
**202 MB** checkpoints (`models/*_100k_seed_*.zip`).
- **Consequence:** 2.2 GB of checkpoints, slow training, wasted capacity on a 32×32 field.
- **Fix:** add pooling / strided convs / global-average-pool → ~100× smaller heads, faster, and
  the checkpoints become git/HF-friendly.

### A5 — O(n²) Python double-loop spread 🟠
Spread is a nested Python loop over every interior cell every step. At 32×32 this is tolerable; at
the 64×64 / 128×128 the report proposes as "future work," it is quadratically worse and
single-threaded.
- **Fix:** vectorise with NumPy/`scipy.ndimage` convolution or a Numba/torch kernel before any
  resolution scaling.

---

## B. Reproducibility & determinism (verified from source)

### B1 — Environment stochasticity is NOT seeded 🔴
`reset(seed=...)` calls `super().reset(seed=seed)` (which seeds `self.np_random`), but `step()` uses
the **global** `np.random.rand()` for spread. The per-episode `reset(seed=seed+ep)` in the training
eval loop therefore **does not control the dynamics at all** — it is cosmetic. Episodes differ only
because the global NumPy stream advances.
- **Consequence:** evaluation is not reproducible per-episode; "seeded evaluation" is not actually
  seeded; re-running yields different numbers (this is *also* why the CSVs disagree, §B4).
- **Fix:** use `self.np_random.random()` everywhere in `step()`; thread the seed through the env;
  seed SB3 via `set_random_seed`.

### B2 — No held-out scenarios: train and eval use the SAME fixed tensor 🔴🔴
Every env loads one file — `grids/32x32/state_tensor.npy` — as `self.initial_tensor`, and `reset()`
restores exactly that. Training and evaluation occur on the **identical** ignition map, fuel,
wind, terrain. There is **one scenario per region**, used for train *and* test.
- **Consequence:** this is textbook **train/test leakage**. "Generalisation," "robustness," and
  "evaluation" within a region are unmeasured — the policy is tested on precisely what it trained
  on. Burned-cell / reward "results" describe memorised performance on a single map.
- **Fix:** generate a *distribution* of scenarios (randomised ignition points, perturbed
  weather/fuel, multiple ROIs/dates); split into disjoint train/val/test sets; report test-set
  metrics with CIs.

### B3 — "Baseline evaluation" contains no baseline 🔴
`10_saudi_baseline_evaluation.ipynb` loads the Saudi PPO and evaluates it on Saudi. There is **no
random-policy, no no-op, and no heuristic control** anywhere in the repo.
- **Consequence:** the central efficacy claim ("RL learns effective suppression," report §3.1) is
  unsupported — there is no reference point to show the policy does anything (see A3).
- **Fix:** the notebook must include `action = env.action_space.sample()` and a fixed no-op policy
  on identical scenarios; report PPO − baseline.

### B4 — A published results table is partly HARDCODED 🔴🔴
`10_saudi_baseline_evaluation.ipynb`, cell 10: only the Saudi→Saudi row is computed live; the
**Saudi→California (−86674.52) and California→California (−85326.05) rows are hardcoded literal
constants** pasted into the DataFrame, then written to `results/cross_region_evaluation.csv`.
- **Consequence:** `cross_region_evaluation.csv` is a manual collage of numbers from different runs
  with different RNG states. This is exactly why the three files disagree on "CA native":
  `cross_region`=−85,326, `zero_shot`=−82,614, `final_summary`=−86,180. No canonical run exists.
- **Fix:** delete hardcoded rows; compute the full matrix in one script, one RNG regime, one
  metric definition; emit a single canonical results table with run metadata.

### B5 — Metric definitions drift across notebooks 🔴
"Burned cells" uses threshold **`> 0.5`** in the training eval (`05`/`06`) but **`> 0.2`** in the
evaluation notebooks (`07`/`10`). "Fire intensity" is `Σ fire` in some places and the same as
"reward" magnitude in others.
- **Consequence:** burned-cell counts are not comparable across files (e.g. Saudi 521 @0.5 vs 598
  @0.2). Any table mixing them is internally inconsistent.
- **Fix:** define metrics once in `eval/metrics.py`; import everywhere; state thresholds in the
  paper.

### B6 — Transfer uses single legacy models, not the 5-seed ensemble 🟠
`07` and `10` load `ppo_saudi_32x32` / `ppo_california_32x32` (the 65–202 MB **single-run** legacy
checkpoints), **not** the `*_100k_seed_{0..4}` models the multi-seed rigor is built on.
- **Consequence:** the transfer headline rests on one seed each; the "multi-seed reproducibility"
  selling point does not apply to the transfer result.
- **Fix:** evaluate transfer across all 5×5 (train-seed × eval) and report mean ± CI.

### B7 — No dependency pinning, Colab-only execution 🟠
Every notebook starts with `from google.colab import drive; drive.mount(...)` and
`!pip install gymnasium stable-baselines3` (unpinned). Hardcoded `/content/drive/MyDrive/...` paths
(20+ per training notebook).
- **Consequence:** runs only on the author's Drive; library versions float; no lockfile → results
  are not reconstructable on any other machine.
- **Fix:** pin versions in `pyproject.toml`/lockfile; central config-driven paths; Dockerfile.

### B8 — Inconsistent env validation 🟡
`check_env(env)` is called in `06` (California) but not in `05` (Saudi) or the eval notebooks.
- **Fix:** one conformance test in `tests/test_env_api.py`.

---

## C. The headline scientific claim is, as evidenced, invalid 🔴🔴🔴

**Report claim (§22):** "Policies trained under Saudi conditions do not generalise to California …
performance degradation exceeded 100% relative reward difference … strongly supports ecological
specialization."

**How the −100% is obtained:** Saudi-on-Saudi (−40k) vs Saudi-on-California (−86k). That compares
performance across **two different environments with different fire loads** (§A2). It is not a
transfer measurement — it is a fuel-density measurement.

**The correct comparison** is *on the same target environment*: Saudi-policy-on-California vs
California-policy-on-California. From the project's own files:

| Source file | Saudi→CA | CA→CA | Gap |
|---|---|---|---|
| `zero_shot_transfer_results.csv` | −86,174 | −82,614 | **4.3 %** |
| `cross_region_evaluation.csv` | −86,675 | −85,326 | **1.6 %** |

**The Saudi policy performs within ~2–4 % of the native California policy on California.** That is
the *opposite* of "substantial transfer degradation." The most parsimonious explanation, given A3
(agent near-powerless) and B3 (no baseline), is that **both policies perform near the do-nothing
floor**, so transfer looks "free" because neither policy meaningfully controls the fire.

- **Consequence:** the paper's central narrative ("strong ecological specialization / transfer
  failure") is contradicted by the repo's own numbers under the correct comparison. A reviewer will
  reject this outright.
- **Fix:** (1) normalise the metric (A2); (2) add baselines (B3) to establish there is any signal
  to transfer; (3) compute the full symmetric 2×2 matrix incl. **California→Saudi** (currently
  never run); (4) reframe the claim around the *corrected* numbers.

### MARL claim caveat
Report §17 reports "1 agent 0.688 → 5 agents 0.513" and says containment improved ~25%. Values
*decrease*, so the metric is fraction-fire-remaining (lower=better) and 5 agents is ~25% lower —
the claim is directionally defensible **once the metric is labelled**. But `08_multi_agent` trains
for **20k** steps vs **100k** single-agent — budgets are not matched, so cross-condition comparison
is confounded.

---

## D. Cross-region normalisation confound 🟠

Per-channel min–max `normalize()` is applied **independently per region** in each pipeline notebook
(`01/03/04`), then stacked unchanged in `05_*_tensor_stacking`. There is **no shared scaler** across
Saudi and California.
- **Consequence:** a value of 0.5 in Saudi temperature ≠ 0.5 in California temperature in absolute
  terms; absolute climate differences (Saudi far hotter/drier) are normalised away while only
  relative spatial structure survives. Cross-region transfer therefore conflates genuine domain
  shift with an artificial per-region rescaling. The transfer experiment cannot isolate "ecology."
- **Fix:** fit one scaler on a shared reference (or use physical units with documented ranges);
  record scaler parameters in the data manifest; state the normalisation in the data card.

---

## E. Revised assessment scores (post-forensic)

| Dimension | Round-1 | Round-2 (after reading code) | Why it dropped |
|---|---|---|---|
| Reproducibility | 3/10 | **2/10** | Unseeded dynamics (B1), hardcoded results (B4), Colab-only (B7) |
| Publication readiness | 4/10 | **2/10** | Headline claim invalid (C), leakage (B2), no baseline (B3), metric drift (B5) |
| Experimental integrity | — | **2/10** | Hardcoded table rows + memorised single scenario |
| Repository health | 2/10 | 2/10 | unchanged |
| Open-source readiness | 1/10 | 1/10 | unchanged |

**Engineering maturity** (env copied 6×, no module, no tests) and **benchmark validity** (single
scenario, no baseline, leaked eval) are the two lowest-scoring axes and must be fixed before this
can be called a "benchmark."

---

## F. Highest-priority *scientific* fixes (distinct from the repo-engineering fixes in the plan)

1. **Kill evaluation leakage (B2)** — scenario distribution + disjoint train/val/test. *Without
   this, no number in the repo is a generalisation result.*
2. **Seed the dynamics (B1)** — `self.np_random` in `step()`; reproducible episodes.
3. **Add baselines (B3/A3)** — random + no-op; report PPO − baseline.
4. **Normalise the metric (A2)** and recompute; stop comparing raw reward across environments.
5. **Recompute the full 2×2 transfer matrix in one run (B4/B6)** incl. California→Saudi; delete
   hardcoded rows; one metric definition (B5).
6. **Re-evaluate the central claim (C)** against corrected numbers — be prepared to retract or
   reframe "ecological specialization."
7. **Fix or downscope the wind claim (A1)** and document the spread model with equations.
8. **Shared cross-region normalisation (D)**.

These are ordered by scientific impact. Items 1–4 are prerequisites for *any* publishable claim;
5–6 determine whether the paper's thesis stands; 7–8 close credibility gaps a reviewer will probe.

---

*Forensic review based on direct reading of notebook source on 2026-06-26.*
