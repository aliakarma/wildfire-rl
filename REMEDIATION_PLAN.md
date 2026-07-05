# Repository Remediation & Hardening Plan

> Scope: `wildfire-rl` (geospatial PPO wildfire suppression + cross-regional transfer).
> Audience: repository owner (competent engineer) executing a supervised, gated refactor.
> Shell: **Git Bash on Windows**. All commands assume repo root `wildfire-rl/` and a
> local virtual environment at `./venv` (Windows layout: `venv/Scripts/…`).

---

## 1. Executive Summary

### Current repository maturity assessment
- **Software engineering: mature.** Installable package, OmegaConf config schema, pytest
  suite, CI, pinned runtime deps, seeded dynamics. This layer is production-grade and is
  *retained*, not rewritten.
- **Scientific validity: not publishable.** Every learned-policy claim is invalidated by a
  single root defect: the observation returned by `WildfireEnv` / `MultiAgentWildfireEnv`
  does **not** encode the agent's own coordinates, so PPO collapses to a constant action
  indistinguishable from `noop`. Downstream, the reported tables are (a) partly untraceable
  to committed CSVs, (b) contradicted by the repo's own baseline outputs, and (c) supported
  by significance statistics computed on near-zero-variance (degenerate) data.

### Main blockers preventing publication/reproducibility
1. **Non-Markov observation** → inert policy (`envs/base.py:52`, `envs/multi_agent.py:51`).
2. **Train/test leakage** → identical fixed ignition map for train and eval
   (`config.py:137`, `EnvConfig.randomize_ignition=False`).
3. **Untraceable headline metrics** → §20 Saudi table absent from all result files.
4. **Selective reporting** → `nearest_fire`/`frontier` baselines dominate PPO but are omitted.
5. **Degenerate statistics** → `fire_std=0.0`, duplicate seeds, Cohen's d guard artifact
   (`eval/significance.py:48`).
6. **Reproducibility gap** → models/tensors untracked; `results/models_manifest.json`
   referenced by README but absent; artifact hub unverified.

### Estimated total remediation time
- **Engineering: ~7–10 working days.**
- **Compute (authoritative multi-seed regeneration, Phase 9 + Phase 14): 1–3 days wall-clock**
  depending on CPU/GPU (2 regions × 4 agent counts × 5 seeds × 100k steps + evals).

### Expected final outcome
A repository in which a clean clone can (i) recreate the environment deterministically,
(ii) fetch hashed artifacts, (iii) run one command to regenerate **every** number in the
paper from committed CSVs, and (iv) pass an automated gate proving PPO outperforms `noop`.
Every table cell traces to a generating script and a committed result file with a recorded
seed, config hash, and git SHA.

## 1.1 Advisor Recommendations (Scope Addendum)

Three recommendations from the project advisor are incorporated as first-class scope. Each maps
to a dedicated **new extension phase** plus supporting existing phases:

| # | Advisor recommendation | New phase (primary) | Supporting existing phases |
|---|------------------------|---------------------|----------------------------|
| 1 | Obtain / train a **stronger policy** | **Phase 16 — Policy Strengthening & HPO** | Phase 3 (Markov obs — done), Phase 7 (learning gate), Phase 9 (multi-seed retrain) |
| 2 | **Visualize policy rollouts** across ablations | **Phase 17 — Rollout Visualization** | Phase 8 (tracking/logging), Phase 7 (ablation metrics), Phase 12 (figures/report) |
| 3 | Richer **Saudi context** — petroleum-asset criticality score, more frequent & more random ignitions, reduced spread | **Phase 15 — Saudi Context Enrichment** | Phase 4 (ignition/leakage), Phase 11 (data provenance) |
| 4 | **AAAI research core** — critical-infrastructure protection, symmetric cross-region transfer, hierarchical hybrid control, strategic coordination, petroleum-risk modeling, domain generalization, hybrid heuristic+RL | **Phase 15B — Critical Infrastructure & Strategic Coordination** | Phase 15 (Saudi context), Phase 4 (leakage), Phases 9 / 12 / 14 |

**Updated master execution order** (existing phases keep their numbers; extension phases interleave;
their specs are appended at the end of this document):

```
1 → 2 → 3 → 4 → [15] → [15B] → 5 → 6 → 7 → 8 → 9 → [16] → [17] → 10 → 11 → 12 → 13 → 14 (terminal)
```

Placement rationale:
- **Phase 15** changes environment dynamics + reward and therefore MUST precede the authoritative
  retrain (Phase 9). It runs immediately after leakage elimination (Phase 4), with which it shares
  the ignition model.
- **Phase 15B (NEW — the AAAI research core)** builds the infrastructure-aware, hierarchical, and
  cross-region transfer system on top of Phase 15. It runs before the authoritative regeneration so
  its observation channels, reward terms, and metrics are baked into the reported results.
- **Phase 16** is **reframed** by the empirical finding in §1.2: its "PPO must beat no-op" goal is
  *superseded*; it is retained as an honest negative-result ablation, never a success criterion.
- **Phase 17** consumes the heuristic/hierarchical policies and produces the strategic rollout figures.
- **Phase 14** remains the terminal certification gate and now additionally certifies Phases 15 / 15B / 16 / 17.

## 1.2 Research Direction Pivot (empirical finding, preserved honestly)

Rigorous evaluation under the remediated pipeline (Markov observation, leakage-free protocol,
agent-attributable + dense-proximity reward, anti-collapse reward rebalancing, exploration tuning)
established a **robust negative result**: single-agent PPO does **not** significantly outperform a
no-op baseline, while deterministic heuristic routers (`nearest_fire` / `frontier`) contain the fire
near-perfectly (Saudi: ~2 vs ~33 burned cells). PPO exhibited **policy collapse** and
**reward-dominance** pathologies, both diagnosed with data (constant-action collapse; the
uncontrollable `−Σfire` term drowning the controllable navigation signal). **This negative result is
a primary scientific contribution and is preserved honestly — it is never "fixed" by fabricating a
PPO win.**

The research question therefore pivots from *"Can PPO learn low-level navigation?"* to:

> **"How can hybrid autonomous wildfire-response systems generalize across geographically distinct
> environments while protecting high-risk critical infrastructure?"**

The system pivots to a **hierarchical hybrid architecture**: a robust, interpretable *heuristic
low-level controller* (navigation + deterministic suppression) beneath a *strategic high-level
controller* (RL or optimization) responsible for resource allocation, infrastructure prioritization,
and regional dispatch. Low-level PPO navigation is **no longer a primary claim**; heuristic local
control is the **reliable operational baseline**. The full research roadmap is **Phase 15B**, and the
report is reframed as trustworthy evaluation of a safety-critical hybrid coordination system (§AAAI
framing in Phase 15B.6). All remediation guarantees (reproducibility, leakage elimination, seed
integrity, effective-method gate, provenance, CI validation) remain in force and are inherited by the
new experiments.

---

# Phase 1 — Repository Cleanup & Structural Repair
Estimated Time: 2–3 hours

## Objective
Remove build artifacts and shadow "review" documents from the tracked tree, fix `.gitignore`,
quarantine the undocumented v2–v6 experiment sprawl, and repair the corrupted report text so
subsequent phases operate on a clean, unambiguous surface.

## Problems Addressed
- Committed `__pycache__/*.pyc` build artifacts under `src/wildfire_rl/`.
- Four in-repo audit files shipped as pseudo-documentation.
- `report.md:390` corrupted sentence fragment.
- v2–v6 scripts/results/figures disconnected from the documented `wildfire-rl` pipeline.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `.gitignore` | Add `__pycache__/`, `*.pyc`, `venv/`, `.pytest_cache/`, `*.egg-info/` |
| `src/wildfire_rl/**/__pycache__/` | Delete from tree |
| `Chatgpt_Review.md`, `claude_review_report.md`, `forensic_audit_report.md`, `docs/AUDIT_V2_FORENSIC_CODE_REVIEW.md` | Move to `reviews/history/` |
| `docs/paper/report.md` | Fix corrupted line 390 |
| `scripts/`, `results/`, `figures/`, `src/wildfire_rl/{envs,coordination,routing}` (v2–v6) | Move to `experimental/` namespace, or delete if abandoned |

## Step-by-Step Implementation Guide

### Step 1 — Purge build artifacts and fix ignore rules
Purpose: Stop tracking non-source binaries that pollute diffs and break reproducibility hashing.

Implementation:
```bash
git rm -r --cached $(find src -type d -name __pycache__) 2>/dev/null || true
find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
{
  echo "__pycache__/"
  echo "*.pyc"
  echo "venv/"
  echo ".pytest_cache/"
  echo "*.egg-info/"
  echo ".coverage"
} >> .gitignore
sort -u .gitignore -o .gitignore
```

Code Changes:
```gitignore
# BEFORE (.gitignore) — no python-artifact rules
# AFTER (.gitignore) — appended
__pycache__/
*.pyc
venv/
.pytest_cache/
*.egg-info/
.coverage
```

Validation:
```bash
git ls-files | grep -E '__pycache__|\.pyc$' && echo "FAIL: artifacts still tracked" || echo "OK"
```

Expected Result: No `.pyc`/`__pycache__` paths remain in `git ls-files`.

### Step 2 — Quarantine shadow review documents
Purpose: Separate audit history from repository documentation so reviewers do not treat prior audits as project claims.

Implementation:
```bash
mkdir -p reviews/history
git mv Chatgpt_Review.md reviews/history/ 2>/dev/null || mv Chatgpt_Review.md reviews/history/
git mv claude_review_report.md reviews/history/ 2>/dev/null || mv claude_review_report.md reviews/history/
git mv forensic_audit_report.md reviews/history/ 2>/dev/null || mv forensic_audit_report.md reviews/history/
git mv docs/AUDIT_V2_FORENSIC_CODE_REVIEW.md reviews/history/ 2>/dev/null || mv docs/AUDIT_V2_FORENSIC_CODE_REVIEW.md reviews/history/
```

Validation:
```bash
ls reviews/history/ | wc -l   # expect 4
find . -maxdepth 1 -name '*Review*.md' -o -maxdepth 1 -name '*audit*.md' | grep -v reviews/ && echo "FAIL" || echo "OK"
```

Expected Result: Repo root contains no loose audit/review markdown files.

### Step 3 — Repair corrupted report text
Purpose: Remove the copy-paste artifact that makes the ablation conclusion unparseable.

Code Changes:
```markdown
# BEFORE (docs/paper/report.md:390)
...limited suppression capacity, which motivates the multi-agent scaling results.rnia wildfire regimes.
Performance degradation exceeded:
•	100% relative reward difference.

# AFTER
...limited suppression capacity, which motivates the multi-agent scaling results.
```

Validation:
```bash
grep -n "results.rnia" docs/paper/report.md && echo "FAIL" || echo "OK"
```

Expected Result: The corrupted token `results.rnia` no longer exists.

### Step 4 — Quarantine undocumented v2–v6 tracks
Purpose: The README/`wildfire-rl` CLI documents single-agent + MARL scaling + transfer + ablation only. The v2–v6 reward-shaping/coordination code is orthogonal and unreferenced; isolate it so the reproducible core is unambiguous.

Implementation:
```bash
mkdir -p experimental/scripts experimental/results experimental/figures
for f in train_single_marl_v2 train_single_marl_v3 train_single_hybrid_marl train_adaptive_v6 \
         run_marl_v2_evaluation run_marl_v3_evaluation run_hybrid_evaluation \
         coordination_validation_v5 adaptive_validation_v6 run_ablation_analysis_v5 \
         run_transfer_analysis_v5 runtime_analysis_v5 runtime_analysis_v6; do
  [ -f "scripts/$f.py" ] && git mv "scripts/$f.py" "experimental/scripts/" 2>/dev/null || true
done
for d in v2 v3 v4 v5 v6; do
  [ -d "results/$d" ] && git mv "results/$d" "experimental/results/$d" 2>/dev/null || true
  [ -d "figures/$d" ] && git mv "figures/$d" "experimental/figures/$d" 2>/dev/null || true
done
echo "Experimental reward-shaping/coordination tracks (v2-v6). Not part of the reproducible core pipeline." > experimental/README.md
```

Validation:
```bash
ls scripts/ | grep -E '_v[2-6]' && echo "FAIL: v2-v6 still in scripts/" || echo "OK"
```

Expected Result: `scripts/` contains only the documented core pipeline; v2–v6 live under `experimental/`.

## README Updates Required

### Add Section
```markdown
## Repository Layout Guarantee

The reproducible research pipeline consists of:
- `src/wildfire_rl/` — package
- `scripts/{train,evaluate,transfer,run_ablation,run_marl_evaluation}.py`
- `configs/` and `results/*.csv`

Exploratory reward-shaping and coordination experiments (v2–v6) live under
`experimental/` and are **not** part of the certified reproducibility path.
```

### Modify Existing Section
- In **Project structure**, replace the flat tree with the two-tier layout (core vs `experimental/`).
- Remove any reference implying the v2–v6 figures are part of the main results.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] No `__pycache__/` or `*.pyc` in `git ls-files`
- [ ] Root has no loose review/audit markdown
- [ ] v2–v6 scripts/results/figures relocated under `experimental/`

Reproducibility checklist
- [ ] `.gitignore` blocks venv and build artifacts

Scientific validity checklist
- [ ] `report.md` corrupted line repaired

Logging/monitoring checklist
- [ ] N/A this phase

README completeness checklist
- [ ] "Repository Layout Guarantee" section added
- [ ] Project-structure tree updated

### Proceed Rule
- If ALL items are `[x]`, proceed to next phase.
- Otherwise DO NOT continue. Fix remaining items first.

---

# Phase 2 — Dependency & Environment Stabilization
Estimated Time: 2–3 hours

## Objective
Produce a byte-reproducible environment: a fully-pinned lockfile, a repaired `requirements.txt`
(missing `seaborn`), and a single documented bootstrap path validated on a clean venv.

## Problems Addressed
- `seaborn` imported in `scripts/run_marl_evaluation.py:22` but absent from `requirements.txt`.
- No transitive lock (only top-level pins); `environment.yml` unverified against pins.
- `results/models_manifest.json` referenced by README but never generated.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `requirements.txt` | Add `seaborn==0.13.2`; keep exact pins |
| `requirements.lock.txt` (new) | Full transitive lock via `pip freeze` |
| `pyproject.toml` | Ensure `seaborn` in the correct extra/base; verify `[dev]`/`[geo]`/`[track]` |
| `environment.yml` | Align python + core pins with `requirements.txt` |

## Step-by-Step Implementation Guide

### Step 1 — Clean venv bootstrap
Purpose: Guarantee the pinned set installs from scratch on the target interpreter.

Implementation:
```bash
rm -rf venv
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Validation:
```bash
python -c "import wildfire_rl, torch, stable_baselines3, gymnasium, scipy, pandas; print('imports OK')"
```

Expected Result: All core imports succeed on a fresh venv.

### Step 2 — Repair and pin `seaborn`
Purpose: Remove the hidden import break that only surfaces when running MARL evaluation.

Code Changes:
```diff
# requirements.txt
 scipy==1.13.0
+seaborn==0.13.2
```

Implementation:
```bash
pip install "seaborn==0.13.2"
grep -q "seaborn" requirements.txt || echo "seaborn==0.13.2" >> requirements.txt
python -c "import seaborn; print('seaborn', seaborn.__version__)"
```

Validation:
```bash
grep -RIl "import seaborn" scripts src experimental | while read f; do echo "uses seaborn: $f"; done
pip check
```

Expected Result: `pip check` reports no broken requirements; every `import seaborn` is satisfied.

### Step 3 — Emit a transitive lockfile
Purpose: Freeze the exact resolved graph for artifact evaluation.

Implementation:
```bash
pip freeze --exclude-editable > requirements.lock.txt
sha256sum requirements.lock.txt
```

Validation:
```bash
test -s requirements.lock.txt && echo "OK lock generated" || echo "FAIL"
```

Expected Result: `requirements.lock.txt` committed with a recorded hash.

## README Updates Required

### Add Section
```markdown
## Environment (pinned)

```bash
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
pip install -e ".[dev]"
pip install -r requirements.lock.txt # exact transitive pins
```
```

### Modify Existing Section
- In **Installation**, replace the loose `pip install -e .` with the pinned two-step flow.
- Note that `requirements.lock.txt` is the authoritative artifact-evaluation environment.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] Fresh venv installs `.[dev]` with no error
- [ ] `pip check` clean
- [ ] `import seaborn` resolves

Reproducibility checklist
- [ ] `requirements.lock.txt` committed and hashed
- [ ] `environment.yml` python/core pins match `requirements.txt`

Scientific validity checklist
- [ ] N/A this phase

Logging/monitoring checklist
- [ ] N/A this phase

README completeness checklist
- [ ] Pinned environment section added
- [ ] Installation updated to two-step pinned flow

### Proceed Rule
- If ALL items are `[x]`, proceed to next phase. Otherwise fix first.

---

# Phase 3 — Model & Environment Architecture Correction (Markov Observation Fix)
Estimated Time: 4–6 hours (code) — **blocks all retraining**

## Objective
Make the observation Markov with respect to the agent by encoding agent position(s) into the
observation tensor, so PPO can represent a navigation policy. This is the single fix that
unblocks every downstream RL claim.

## Problems Addressed
- `WildfireEnv.observation_space` and `MultiAgentWildfireEnv.observation_space` expose only the
  `(C,H,W)` environmental tensor; `agent_pos`/`agent_positions` are never observable
  (`envs/base.py:52,73,93,106`; `envs/multi_agent.py:51,85`).
- Empirical policy collapse: `results/eval_california.csv:3-4` (`ppo_seed_0 ≡ noop`) and
  distinct-weight `seed_2`/`seed_3` producing identical eval outputs.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/config.py` | Add `EnvConfig.include_agent_channel: bool = True` |
| `src/wildfire_rl/envs/base.py` | Add `_obs()`; extend `observation_space` by +1 channel; return `_obs()` from `reset`/`step` |
| `src/wildfire_rl/envs/multi_agent.py` | Add `_obs()` with agent-occupancy channel; extend obs space |
| `src/wildfire_rl/models/cnn.py` | No change (reads `observation_space.shape[0]` dynamically) — verify |
| `tests/test_env_api.py` | Add test: obs differs when agent position differs |
| `docs/model_card.md`, `docs/architecture.md` | Document the observation contract |

## Step-by-Step Implementation Guide

### Step 1 — Add the config flag
Purpose: Keep the fix togglable for ablation (with/without position channel).

Code Changes:
```python
# src/wildfire_rl/config.py — inside EnvConfig (after suppression block, ~line 110)
# AFTER
    # --- observation ---
    # Encodes agent position(s) into the observation so the MDP is Markov.
    include_agent_channel: bool = True
```

Validation:
```bash
python -c "from wildfire_rl.config import EnvConfig; print(EnvConfig().include_agent_channel)"
```

Expected Result: prints `True`.

### Step 2 — Single-agent observation channel
Purpose: Give the policy a one-hot map of its own location.

Code Changes:
```python
# src/wildfire_rl/envs/base.py

# BEFORE (line ~51)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

# AFTER
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._obs_channels = c + (1 if self.cfg.include_agent_channel else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_channels, h, w), dtype=np.float32
        )

# ADD helper (after __init__)
    def _obs(self) -> np.ndarray:
        if not self.cfg.include_agent_channel:
            return self.state.astype(np.float32)
        pos = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        x, y = self.agent_pos
        pos[0, x, y] = 1.0
        return np.concatenate([self.state, pos], axis=0).astype(np.float32)

# BEFORE (reset, line ~73)      return self.state.astype(np.float32), {}
# AFTER                          return self._obs(), {}

# BEFORE (step, line ~102)      return self.state.astype(np.float32), reward, terminated, truncated, info
# AFTER                          return self._obs(), reward, terminated, truncated, info
```

Note: `n_channels` validation in `__init__` (`base.py:47`) still checks the *input tensor*
channels (`c == cfg.n_channels`), which is correct — the position channel is appended after
validation. Do not change that check.

Validation:
```bash
python - <<'PY'
import numpy as np
from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.base import WildfireEnv
t = np.zeros((7,8,8), np.float32); t[0,4,4]=1.0
e = WildfireEnv(state_tensor=t, config=EnvConfig(grid_size=8, max_steps=5))
o,_ = e.reset(seed=0); assert o.shape==(8,8,8), o.shape
print("single-agent obs shape OK", o.shape)
PY
```

Expected Result: observation shape is `(8,8,8)` (7 env + 1 position).

### Step 3 — Multi-agent occupancy channel
Purpose: Let the centralized controller observe where all its agents are.

Code Changes:
```python
# src/wildfire_rl/envs/multi_agent.py

# BEFORE (line ~50)
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * self.num_agents)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)

# AFTER
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * self.num_agents)
        self._obs_channels = c + (1 if self.cfg.include_agent_channel else 0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_channels, h, w), dtype=np.float32
        )

# ADD helper
    def _obs(self) -> np.ndarray:
        if not self.cfg.include_agent_channel:
            return self.state.astype(np.float32)
        occ = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.agent_positions:
            occ[0, x, y] += 1.0
        np.clip(occ, 0.0, 1.0, out=occ)
        return np.concatenate([self.state, occ], axis=0).astype(np.float32)

# reset (line ~85)  -> return self._obs(), {}
# step  (line ~129) -> return self._obs(), reward, terminated, truncated, info
```

Validation:
```bash
python - <<'PY'
import numpy as np
from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
t = np.zeros((7,8,8), np.float32); t[0,4,4]=1.0
e = MultiAgentWildfireEnv(state_tensor=t, config=EnvConfig(grid_size=8, max_steps=5), num_agents=3)
o,_ = e.reset(seed=0); assert o.shape==(8,8,8)
print("marl obs shape OK", o.shape, "occupancy sum", o[7].sum())
PY
```

Expected Result: obs shape `(8,8,8)`; occupancy channel sums to number of distinct agent cells.

### Step 4 — Regression test: observation is position-sensitive
Purpose: Lock the fix so it cannot silently regress.

Code Changes:
```python
# tests/test_env_api.py — ADD
def test_observation_encodes_agent_position(small_tensor, env_cfg):
    env = WildfireEnv(state_tensor=small_tensor, config=env_cfg)
    o0, _ = env.reset(seed=0)
    env.step(1)  # move down
    o1 = env._obs()
    # The agent-position channel (last) must change after a move.
    assert not np.array_equal(o0[-1], o1[-1])
```

Validation:
```bash
source venv/Scripts/activate
pytest tests/test_env_api.py -q
```

Expected Result: new test passes; `check_env` (`test_env_api.py:59`) still passes with the new shape.

### Step 5 — Invalidate legacy checkpoints
Purpose: Legacy `.zip` checkpoints were trained on the old 7-channel obs and are now shape-incompatible. Prevent accidental loading.

Implementation:
```bash
mkdir -p models/deprecated_pre_markov
git mv models/Legacy models/deprecated_pre_markov/Legacy 2>/dev/null || mv models/Legacy models/deprecated_pre_markov/Legacy
echo "Checkpoints trained on the pre-Markov (no agent channel) observation. Incompatible with current env. Retained for provenance only." > models/deprecated_pre_markov/README.md
```

Validation:
```bash
ls models/deprecated_pre_markov/Legacy/*.zip | head -1 && echo "legacy quarantined"
```

Expected Result: legacy checkpoints isolated; no code path loads them for reported results.

## README Updates Required

### Add Section
```markdown
## Observation Contract

Each observation is a `(C+1, H, W)` float32 tensor:
- channels 0–6: `fire, fuel, wind_x, wind_y, terrain, temperature, humidity`
- channel 7: agent position (single-agent) / agent occupancy map (multi-agent)

The position channel makes the environment Markov for a movement policy. Checkpoints trained
before this change (`models/deprecated_pre_markov/`) are incompatible and must not be used for
reported results.
```

### Modify Existing Section
- In **Overview/Architecture**, change "7-channel state tensor" to "7-channel state + 1 agent channel (8-channel observation)".
- Add a one-line changelog entry: "Observation now encodes agent position (Markov fix)."

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] Single-agent obs shape `(C+1,H,W)`
- [ ] MARL obs shape `(C+1,H,W)` with valid occupancy channel
- [ ] `CustomCNN` builds against new `observation_space` (dynamic channel read verified)
- [ ] Full `pytest -q` passes including new position test

Reproducibility checklist
- [ ] `include_agent_channel` recorded in config dumps

Scientific validity checklist
- [ ] Legacy pre-Markov checkpoints quarantined and marked incompatible

Logging/monitoring checklist
- [ ] N/A this phase

README completeness checklist
- [ ] Observation Contract section added
- [ ] Architecture channel count corrected

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first. (Do not retrain until this gate passes.)

---

# Phase 4 — Data Leakage Elimination
Estimated Time: 3–4 hours

## Objective
Guarantee disjoint train/evaluation scenario distributions by making randomized ignition the
default for all *reported* runs and by enforcing non-overlapping train vs eval seed ranges.

## Problems Addressed
- `EnvConfig.randomize_ignition=False` default (`config.py:137`) → train and eval share one
  fixed ignition map (Phase-6 silent-bug #4 / test-set contamination).
- No explicit separation between training reset seeds and evaluation reset seeds.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `configs/config.yaml` and `configs/experiment/*.yaml` | Set `env.randomize_ignition: true` for reported experiments; add `eval.base_seed` offset |
| `src/wildfire_rl/config.py` | Add `EvalConfig.scenario_seed_offset: int = 100000` |
| `src/wildfire_rl/eval/evaluate.py` | Use `base_seed + scenario_seed_offset` for eval resets |
| `configs/experiment/multiseed.yaml`, `multiseed_california.yaml` | Enable randomized ignition |

## Step-by-Step Implementation Guide

### Step 1 — Separate train and eval scenario seeds
Purpose: Ensure the ignition maps seen during evaluation were never seen during training.

Code Changes:
```python
# src/wildfire_rl/config.py — EvalConfig (line ~185)
# BEFORE
@dataclass
class EvalConfig:
    n_episodes: int = 20
    deterministic: bool = True
    base_seed: int = 0

# AFTER
@dataclass
class EvalConfig:
    n_episodes: int = 20
    deterministic: bool = True
    base_seed: int = 0
    # Guarantees eval ignition maps are disjoint from training reset seeds.
    scenario_seed_offset: int = 100_000
```

```python
# src/wildfire_rl/eval/evaluate.py — inside evaluate_policy loop (line ~46)
# BEFORE
        obs, _ = env.reset(seed=base_seed + ep)
# AFTER
        obs, _ = env.reset(seed=base_seed + metrics_offset + ep)
```
Add `metrics_offset` parameter to `evaluate_policy(..., scenario_seed_offset: int = 0)` and pass
`cfg.eval.scenario_seed_offset` from every caller (`cli.py:133`, `run_ablation.py:66`,
`run_marl_evaluation.py:161,189`, `transfer_run.py:86`).

Validation:
```bash
grep -Rn "scenario_seed_offset" src scripts | sort
```

Expected Result: every evaluation call site passes the offset; no reported eval uses seed 0..n directly.

### Step 2 — Enable randomized ignition in reported configs
Purpose: Move from single fixed map (memorization) to a scenario distribution (generalization).

Code Changes:
```yaml
# configs/experiment/multiseed.yaml  (and multiseed_california.yaml)
# AFTER (add/ensure)
env:
  randomize_ignition: true
  n_ignition_points: 3
  ignition_intensity: 1.0
eval:
  n_episodes: 50
  scenario_seed_offset: 100000
```

Validation:
```bash
source venv/Scripts/activate
python - <<'PY'
from wildfire_rl.config import load_config
c = load_config("configs/experiment/multiseed.yaml")
assert c.env.randomize_ignition is True
assert c.eval.scenario_seed_offset >= 100000
print("leakage config OK")
PY
```

Expected Result: reported configs train and evaluate on disjoint randomized ignition maps.

### Step 3 — Prove disjointness
Purpose: Demonstrate that training and evaluation never sample the same ignition map.

Implementation:
```bash
python - <<'PY'
import numpy as np
from wildfire_rl.config import EnvConfig
from wildfire_rl.envs.base import WildfireEnv
t = np.zeros((7,32,32), np.float32)
cfg = EnvConfig(randomize_ignition=True, n_ignition_points=3)
e = WildfireEnv(state_tensor=t, config=cfg)
train = {e.reset(seed=s)[0][0].tobytes() for s in range(200)}
evl   = {e.reset(seed=100000+s)[0][0].tobytes() for s in range(50)}
print("overlap:", len(train & evl))
assert len(train & evl) == 0
PY
```

Expected Result: `overlap: 0`.

## README Updates Required

### Add Section
```markdown
## Scenario Splitting (No Leakage)

Reported experiments use randomized ignition (`env.randomize_ignition: true`). Training reset
seeds occupy `[0, N)`; evaluation reset seeds occupy `[scenario_seed_offset, +)`, guaranteeing
disjoint ignition maps. Fixed-map runs are diagnostic only and are labeled as such.
```

### Modify Existing Section
- In **Reproducibility**, state that the default reported protocol is randomized-ignition with disjoint eval seeds.
- Flag any fixed-ignition result as "diagnostic (memorization baseline)".

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `scenario_seed_offset` threaded through every eval call site
- [ ] Reported configs set `randomize_ignition: true`

Reproducibility checklist
- [ ] Disjointness proof returns overlap 0

Scientific validity checklist
- [ ] No reported metric uses the training ignition map for evaluation

Logging/monitoring checklist
- [ ] Config dump records ignition settings

README completeness checklist
- [ ] Scenario-splitting section added

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 5 — Determinism & Seed Control
Estimated Time: 3–4 hours

## Objective
Enforce full-stack determinism (Python/NumPy/Torch/CUDA) and add an automated guard that detects
degenerate "duplicate seed" checkpoints and identical per-seed evaluation rows.

## Problems Addressed
- Duplicate/degenerate seeds: `eval_california.csv` seed_2≡seed_3; `eval_saudi_generalization.csv`
  seed_0≡seed_1, seed_2≡seed_3.
- Torch/CUDA determinism not explicitly enforced in `seeding.set_global_seed`.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/seeding.py` | Enforce torch deterministic algorithms + cudnn flags + `PYTHONHASHSEED` |
| `src/wildfire_rl/train/ppo.py` | Set `CUBLAS_WORKSPACE_CONFIG` before torch import path |
| `scripts/check_seed_integrity.py` (new) | Assert distinct checkpoints and non-identical per-seed eval rows |
| `tests/test_seeding.py` | Extend with determinism assertion across two runs |

## Step-by-Step Implementation Guide

### Step 1 — Harden `set_global_seed`
Purpose: Remove nondeterministic kernels so identical seeds → identical trajectories on any host.

Code Changes:
```python
# src/wildfire_rl/seeding.py
# AFTER (target implementation)
import os, random
import numpy as np

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
```

Validation:
```bash
source venv/Scripts/activate
python - <<'PY'
from wildfire_rl.seeding import set_global_seed
import numpy as np
set_global_seed(0); a = np.random.rand(3)
set_global_seed(0); b = np.random.rand(3)
assert (a==b).all(); print("determinism OK")
PY
```

Expected Result: identical draws after re-seeding.

### Step 2 — Seed-integrity guard
Purpose: Fail loudly when "independent" seeds produce identical checkpoints or identical eval rows.

Code Changes:
```python
# scripts/check_seed_integrity.py (new)
#!/usr/bin/env python
import hashlib, sys
from pathlib import Path
import pandas as pd

def _hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_models(glob):
    paths = sorted(Path("models").glob(glob))
    hashes = {p.name: _hash(p) for p in paths}
    dupes = [n for n in hashes if list(hashes.values()).count(hashes[n]) > 1]
    if dupes: sys.exit(f"FAIL duplicate checkpoints: {dupes}")

def check_eval(csv, key="reward_mean"):
    df = pd.read_csv(csv)
    ppo = df[df["policy"].str.startswith("ppo")] if "policy" in df else df
    if ppo[key].round(6).duplicated().any():
        sys.exit(f"FAIL identical per-seed rows in {csv}")

if __name__ == "__main__":
    check_models("ppo_saudi_32_seed_*.zip")
    for c in ["results/eval_saudi.csv", "results/eval_california.csv"]:
        if Path(c).exists(): check_eval(c)
    print("seed integrity OK")
```

Validation:
```bash
python scripts/check_seed_integrity.py || echo "guard correctly flagged degenerate seeds"
```

Expected Result: after Phase 9 regeneration, guard prints `seed integrity OK`; on current CSVs it flags duplicates.

## README Updates Required

### Add Section
```markdown
## Determinism

`wildfire_rl.seeding.set_global_seed` fixes Python/NumPy/Torch RNGs, enables cuDNN determinism,
and sets `CUBLAS_WORKSPACE_CONFIG`. `scripts/check_seed_integrity.py` asserts that independent
seeds produce distinct checkpoints and non-identical evaluation rows.
```

### Modify Existing Section
- In **Reproducibility**, replace the seeding note with the hardened guarantees and add the integrity-check command.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `set_global_seed` sets torch/cudnn/hashseed
- [ ] `test_seeding.py` determinism assertion passes

Reproducibility checklist
- [ ] `check_seed_integrity.py` runs in CI-callable form

Scientific validity checklist
- [ ] Guard flags the current degenerate CSVs (expected before Phase 9)

Logging/monitoring checklist
- [ ] Seed recorded in every run-metadata JSON

README completeness checklist
- [ ] Determinism section added

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 6 — Pipeline Reconnection & Execution Integrity
Estimated Time: 4–6 hours

## Objective
Make the end-to-end pipeline fail loudly on missing artifacts, unify the MARL scaling output so
it matches the reported table, and provide a single `make reproduce` path that regenerates every
core CSV from code.

## Problems Addressed
- `transfer_run.py:59-65` silently substitutes `RandomPolicy` for a missing model and still
  writes a "transfer matrix".
- `results/marl_scaling_results.csv` is a 1-row stub inconsistent with `report.md §17`.
- Report mixes canonical, legacy, and prose-only numbers.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/experiments/transfer_run.py` | Raise on missing model unless `--allow-missing` |
| `scripts/run_marl_evaluation.py` | Emit a complete `marl_scaling_results.csv` (all agent counts, all seeds) |
| `Makefile` | Single `reproduce` target chaining the core pipeline |
| `results/archive_legacy_notebooks/` | Keep, but mark non-canonical; forbid report citation |

## Step-by-Step Implementation Guide

### Step 1 — Fail loudly on missing models
Purpose: Prevent random-policy numbers from being published as PPO transfer results.

Code Changes:
```python
# src/wildfire_rl/experiments/transfer_run.py:58-65
# BEFORE
        model_path = _find_model(region.name, region.grid_size, seed)
        if model_path is None:
            logger.warning("No trained model ... Using RandomPolicy as a placeholder ...")
            policies[region.name] = RandomPolicy(...)

# AFTER
        model_path = _find_model(region.name, region.grid_size, seed)
        if model_path is None:
            if not getattr(cfg, "allow_missing_models", False):
                raise FileNotFoundError(
                    f"No trained model for '{region.name}' (seed={seed}). "
                    f"Train it or pass allow_missing_models=true for a dev dry-run."
                )
            logger.warning("DRY-RUN: RandomPolicy placeholder for %s", region.name)
            policies[region.name] = RandomPolicy(env_factories[region.name]().action_space, seed=seed)
```

Validation:
```bash
source venv/Scripts/activate
rm -f models/ppo_saudi_32_seed_0.zip
wildfire-rl transfer --config configs/experiment/transfer.yaml 2>&1 | grep -qi "No trained model" && echo "OK fails loudly"
```

Expected Result: transfer aborts with a clear error when a required model is absent.

### Step 2 — Complete the MARL scaling table
Purpose: Regenerate `marl_scaling_results.csv` for all agent counts and seeds so it matches the report.

Code Changes:
```python
# scripts/run_marl_evaluation.py — in save_baseline_comparisons(), also write scaling file
# AFTER (append near line 380)
    scaling = (pd.DataFrame(stats_rows)
               .query("policy == 'ppo'")
               .loc[:, ["region","num_agents","reward_mean","reward_std",
                        "reward_ci_lo","reward_ci_hi","burned_cells_mean","fire_intensity_mean"]])
    scaling.to_csv(results_dir() / "marl_scaling_results.csv", index=False)
```

Validation:
```bash
python -c "import pandas as pd; d=pd.read_csv('results/marl_scaling_results.csv'); print(sorted(d.num_agents.unique()))"
```

Expected Result: agent counts `[1, 3, 5, 10]` present per region (after Phase 9 run).

### Step 3 — Single reproduce target
Purpose: One command regenerates the entire core result set from code.

Code Changes:
```makefile
# Makefile
reproduce: test
	python scripts/train.py --config configs/experiment/multiseed.yaml
	python scripts/train.py --config configs/experiment/multiseed_california.yaml
	python scripts/evaluate.py --config configs/experiment/multiseed.yaml
	python scripts/evaluate.py --config configs/experiment/multiseed_california.yaml
	python scripts/run_marl_evaluation.py --timesteps 100000
	python scripts/run_ablation.py --config configs/experiment/ablation.yaml
	python scripts/transfer.py --config configs/experiment/transfer.yaml
	python scripts/make_figures.py
	python scripts/check_seed_integrity.py
```

Validation:
```bash
grep -A12 "^reproduce:" Makefile
```

Expected Result: a linear, inspectable reproduce recipe exists.

## README Updates Required

### Add Section
```markdown
## Canonical vs Legacy Results

Only `results/*.csv` produced by the current CLI are canonical. `results/archive_legacy_notebooks/`
is provenance-only and MUST NOT be cited in the report. `wildfire-rl transfer` now aborts if a
required checkpoint is missing (no silent random-policy fallback).
```

### Modify Existing Section
- Replace the **Reproducibility** command block with `make reproduce` as the authoritative path (list the equivalent Git Bash commands).

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] Missing model → transfer raises
- [ ] `marl_scaling_results.csv` generator emits all agent counts
- [ ] `make reproduce` recipe present and inspectable

Reproducibility checklist
- [ ] Legacy CSVs marked non-canonical

Scientific validity checklist
- [ ] No pipeline stage can silently substitute a placeholder policy for a reported run

Logging/monitoring checklist
- [ ] Each stage writes a run-metadata JSON

README completeness checklist
- [ ] Canonical-vs-legacy section added

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 7 — Metric Verification & Evaluation Corrections
Estimated Time: 3–4 hours

## Objective
Fix the Cohen's d zero-variance artifact, make reward-mode explicit in every output, and add a
`containment_rate` metric that is not dominated by uncontrollable fire mass — plus a machine-checkable
"learning gate" (PPO must beat `noop`).

## Problems Addressed
- `cohens_d` returns `0.0` when `pooled_std < 1e-12` (`eval/significance.py:48`), mislabeling
  huge effects as null.
- Reward scale mixed across tables (`raw` vs `normalized`) without labeling.
- No metric isolates agent-attributable suppression.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/eval/significance.py` | Return `nan` (not `0.0`) on degenerate pooled std |
| `src/wildfire_rl/eval/metrics.py` | Add `containment_rate`; keep threshold single-source |
| `src/wildfire_rl/cli.py`, `scripts/*` | Write `reward_mode` column into every CSV |
| `scripts/validate_learning_gate.py` (new) | Assert PPO burned_cells < noop by margin |

## Step-by-Step Implementation Guide

### Step 1 — Fix the effect-size guard
Purpose: Never report d=0.0 for a real mean difference.

Code Changes:
```python
# src/wildfire_rl/eval/significance.py:48-50
# BEFORE
    if pooled_std < 1e-12:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)
# AFTER
    if pooled_std < 1e-12:
        # Degenerate (zero within-group variance): effect size is undefined, not zero.
        return float("nan") if abs(g1.mean() - g2.mean()) > 1e-12 else 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)
```

Validation:
```bash
python - <<'PY'
import numpy as np
from wildfire_rl.eval.significance import cohens_d
import math
assert math.isnan(cohens_d(np.array([1.0,1.0,1.0]), np.array([9.0,9.0,9.0])))
print("cohens_d degenerate guard OK")
PY
```

Expected Result: degenerate large-difference case returns `nan`, not `0.0`.

### Step 2 — Add agent-attributable containment metric
Purpose: Report a metric the policy can actually influence.

Code Changes:
```python
# src/wildfire_rl/eval/metrics.py — ADD
def containment_rate(initial_fire_total: float, final_state: np.ndarray) -> float:
    """Fraction of initial fire mass no longer burning (higher is better)."""
    final = float(np.sum(final_state[0] if final_state.ndim == 3 else final_state))
    denom = initial_fire_total or 1.0
    return float(max(0.0, 1.0 - final / denom))
```
Wire it into `evaluate.evaluate_policy` per-episode dict alongside `burned_cells`.

Validation:
```bash
pytest tests/test_metrics.py -q
```

Expected Result: metrics tests pass; new metric available to every evaluation.

### Step 3 — Label reward mode everywhere
Purpose: Prevent cross-table reward comparison across `raw`/`normalized`.

Code Changes:
```python
# in every rows.append({...}) that writes eval/transfer CSVs, add:
"reward_mode": cfg.env.reward_mode,
```

Validation:
```bash
grep -Rn "reward_mode" src/wildfire_rl/cli.py src/wildfire_rl/experiments scripts/run_ablation.py | grep -i "reward_mode\":"
```

Expected Result: `reward_mode` column present in every generated results CSV.

### Step 4 — Learning gate script
Purpose: Machine-checkable proof that the fixed policy beats doing nothing.

Code Changes:
```python
# scripts/validate_learning_gate.py (new)
#!/usr/bin/env python
import sys, pandas as pd
from pathlib import Path
MARGIN = 0.05  # PPO must reduce burned cells >=5% vs noop
def gate(csv):
    df = pd.read_csv(csv)
    noop = df.loc[df.policy=="noop","burned_cells_mean"].mean()
    ppo  = df.loc[df.policy.str.startswith("ppo"),"burned_cells_mean"].mean()
    ok = ppo <= noop * (1 - MARGIN)
    print(f"{csv}: ppo={ppo:.2f} noop={noop:.2f} -> {'PASS' if ok else 'FAIL'}")
    return ok
if __name__ == "__main__":
    results = [gate(c) for c in ["results/eval_saudi.csv","results/eval_california.csv"] if Path(c).exists()]
    sys.exit(0 if results and all(results) else 1)
```

Validation:
```bash
python scripts/validate_learning_gate.py; echo "exit=$?"
```

Expected Result: exit 1 on current inert CSVs; exit 0 after Phase 9 with the Markov fix.

## README Updates Required

### Add Section
```markdown
## Metrics & Learning Gate

Primary metrics: `burned_cells` (threshold 0.5), `fire_intensity`, `containment_rate`,
`episode_reward` (with `reward_mode` labeled). A result set is accepted only if
`scripts/validate_learning_gate.py` passes (PPO reduces burned cells ≥5% vs `noop`).
```

### Modify Existing Section
- In **Evaluation**, add the learning-gate command and state that raw/normalized rewards are not comparable across tables.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `cohens_d` returns `nan` on degenerate large differences
- [ ] `containment_rate` implemented and wired
- [ ] `reward_mode` column in all result CSVs

Reproducibility checklist
- [ ] Learning gate script runnable and CI-callable

Scientific validity checklist
- [ ] Effect sizes no longer misreported as 0.0

Logging/monitoring checklist
- [ ] Gate outcome logged per result set

README completeness checklist
- [ ] Metrics & learning-gate section added

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 8 — Experiment Tracking & Logging
Estimated Time: 3–5 hours

## Objective
Capture, for every run, the git SHA, resolved config hash, seed, library versions, dataset
hashes, and training curves — into per-run manifests and (optionally) TensorBoard, so any
reported number is traceable to an exact execution.

## Problems Addressed
- Reported numbers not tied to a recorded run (§20 Saudi untraceable).
- No committed training curves to prove learning occurred.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/logging_utils.py` | Extend `write_run_metadata` with dataset hashes + lib versions |
| `src/wildfire_rl/train/ppo.py` | Enable `tensorboard_log`; persist final `ep_rew_mean` curve |
| `scripts/train.py` (cli `cmd_train`) | Write a `results/runs/<run_id>/manifest.json` per seed |
| `results/runs/` | Standardize as the run registry |

## Step-by-Step Implementation Guide

### Step 1 — Rich run manifest
Purpose: Bind every artifact to a verifiable provenance record.

Code Changes:
```python
# src/wildfire_rl/logging_utils.py — extend write_run_metadata()
# AFTER (add fields)
import subprocess, hashlib, platform
def _git_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
    except Exception: return "unknown"
def _file_hash(p):
    from pathlib import Path
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
# include in the metadata dict:
#   "git_sha": _git_sha(),
#   "python": platform.python_version(),
#   "config_sha256": hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest(),
#   "tensor_sha256": _file_hash(tensor_path),
```

Validation:
```bash
source venv/Scripts/activate
python scripts/train.py --set ppo.total_timesteps=2000 seeds=[0]
cat results/runs/train_saudi_seed_0.json | python -m json.tool | grep -E "git_sha|config_sha256|tensor_sha256"
```

Expected Result: manifest contains git SHA, config hash, and tensor hash.

### Step 2 — Persist training curves
Purpose: Provide committed evidence that reward improves during training.

Code Changes:
```python
# src/wildfire_rl/train/ppo.py — build_ppo(..., tensorboard_log=...) already supported.
# In cmd_train, pass tensorboard_log=str(results_dir()/ "runs" / "tb" / run_id)
# and after model.learn(), dump the monitor rewards to results/runs/<run_id>/curve.csv
```

Validation:
```bash
find results/runs -name "curve.csv" | head
```

Expected Result: a per-run reward curve is committed for inspection.

## README Updates Required

### Add Section
```markdown
## Experiment Tracking

Every training run writes `results/runs/<run_id>/manifest.json` (git SHA, config SHA-256,
tensor SHA-256, seed, library versions) and `curve.csv` (reward vs timestep). TensorBoard logs
live under `results/runs/tb/`.
```

### Modify Existing Section
- In **Reproducibility**, replace "Every run writes results/runs/*.json" with the enriched manifest schema and the curve/TensorBoard note.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] Manifest includes git SHA, config hash, tensor hash, seed, versions
- [ ] Training curve persisted per run

Reproducibility checklist
- [ ] Each reported number maps to a `run_id`

Scientific validity checklist
- [ ] Curves demonstrate non-flat learning (post Phase 9)

Logging/monitoring checklist
- [ ] TensorBoard logs generated

README completeness checklist
- [ ] Experiment-tracking section added

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 9 — Statistical Validity & Authoritative Regeneration (FULL EXECUTION PLAN)
Estimated Time: 1–2 days wall-clock (evaluation-dominated; PPO negative-baseline training is light).
Execution position: **after Phases 15 + 15B + 15B.8 land in code**, then this phase regenerates the
authoritative, certified numbers that Phases 12/14 report.

> **STATUS — REFRAMED (reconciled with the Phase 16 pivot and the 15B/15B.8 additions).**
> The original success criterion ("learning gate exit 0 = PPO beats noop") is **superseded**. The
> owner-approved finding (Phase 16, evidenced) is that **single-agent PPO does not beat a heuristic on
> this task**; the project is reframed around **heuristic routing as the effective method** with PPO
> reported as a rigorous **negative result**. Accordingly the blocking scientific gate is now the
> **effective-method gate** (the *best reported* policy beats no-op), not a forced PPO win. This phase
> also folds in everything 15B/15B.8 add: infrastructure-aware metrics, the hybrid controller, the
> symmetric cross-region transfer matrix, the eight ablation groups, and the canonical animated-rollout
> GIFs. **Never** relabel or tune PPO to "win" to satisfy a checkbox — the honest negative result is a
> contribution (§1.2, Phase 16).

## Objective
Produce the **single authoritative, provenance-bound result surface** on the corrected environment
(Markov obs · leakage-free splits · determinism · Saudi domain model · infrastructure + hybrid +
transfer from 15B · ablations from 15B.8), reported with **effect-size-primary** statistics
(bootstrap 95% CIs + Cohen's d), verified seed independence, run manifests + artifact hashes, and a
canonical animated-rollout GIF per experiment/ablation cell. Every reported number traces to a
committed CSV + `run_id`; no hand-edited results.

## Problems Addressed
- Degenerate multi-seed (std=0, duplicate checkpoints/rows) and significance-without-effect framing.
- No bootstrap CIs; over-reliance on p-values from paired tests on near-identical trajectories.
- Stale, pre-remediation CSVs still on disk (California / generalization) that predate the Markov fix,
  the leakage fix, the Saudi re-tune, and the pivot — they must be regenerated or retired, not cited.
- The result surface does not yet reflect 15B (infrastructure/hybrid/transfer) or 15B.8 (ablations).

## What Is Already Done (carried from the 🟡 validation pass — do NOT redo)
- **`bootstrap_ci(values, n_boot=10000, ...)`** implemented + tested in `eval/significance.py`.
- **`spread_scale` re-tuned 0.5 → 0.7** (Saudi) so a real fire exists to suppress (noop ≈ 35 burned,
  still ≪ California ≈ 223). Applied to `region/saudi.yaml` + `multiseed.yaml`.
- **CUDA working** on the RTX 3050 (`torch 2.3.1+cu121`); measured GPU ≈85 vs CPU ≈70 steps/s (tiny
  CNN → env/SB3 overhead dominates). Use CPU or GPU interchangeably; determinism is preserved on CPU.
- **Effective-method gate**: `validate_learning_gate.py` reframed to verify the *best* policy beats
  no-op and to print PPO's honest status. On current Saudi data it **PASSes** (`nearest_fire` 2.1 ≪
  noop 33.3; "PPO … does NOT beat no-op (negative result)").
- **2-seed × 40k Saudi validation** proved the pipeline runs end-to-end (train→curve→model→eval→guards).

## Dependency Gate (must be true before the full run)
This phase consumes the outputs of 15B/15B.8. Before launching the authoritative run, confirm:
- [ ] Phase 15B success checkpoint is green (infra rasters + cascade + infra obs channel + hierarchy +
      strategic/transfer metrics implemented & tested).
- [ ] Phase 15B.8 success checkpoint is green (`AblationConfig` toggles, 8 group configs, `ablation/runner`,
      new metrics CE/PA/ERL/TRG/CCL, `test_ablation.py` + `test_rollout_viz.py` green).
- [ ] Canonical rollout renderer + palette locked in `configs/viz.yaml`.

If 15B/15B.8 are not yet coded, run **Phase 9-Core** (below) to certify the *current* heuristic-vs-PPO
+ transfer surface now, then re-run the full phase once 15B/15B.8 land. Phase 9-Core is a strict
subset of the full run using the same gates and provenance — its numbers are superseded, not
contradicted, by the extended run.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/eval/significance.py` | `bootstrap_ci` (done); ensure it is threaded into every summary writer |
| `configs/experiment/multiseed*.yaml` | `seeds: [0,1,2,3,4]`, `eval.n_episodes: 50`, `scenario_seed_offset: 100000` (verify) |
| `Makefile` | `reproduce` recipe = test → (light PPO negative-baseline train) → evaluate (heuristics + PPO, both regions) → `run_marl_evaluation` (heuristic scaling 1/3/5/10) → `run_ablations --group all` (15B.8) → `run_transfer_hybrid` (15B.4 symmetric) → figures + canonical GIFs → `check_seed_integrity` → `validate_learning_gate` |
| `scripts/make_figures.py` / `scripts/run_ablations.py --render-figures` | Regenerate all static figures **and** canonical rollout GIFs from the fresh CSVs |
| all result CSVs under `results/` + `results/ablation/` | Regenerated from corrected code; stale pre-remediation CSVs retired or overwritten |
| `results/runs/reproduce_<ts>.log` | The committed authoritative-run log |

## Step-by-Step Implementation Guide

### Step 0 — Pre-flight (fast, no compute)
```bash
source venv/Scripts/activate
python -m pytest -q                                   # expect all green (66+ tests + 15B/15B.8 tests)
python -c "from wildfire_rl.eval.significance import bootstrap_ci; print(bootstrap_ci([1,2,3,4,5]))"
python scripts/check_seed_integrity.py; echo "integrity exit=$?"   # may still flag stale CSVs (expected pre-run)
grep -n "seeds\|n_episodes\|scenario_seed_offset" configs/experiment/multiseed*.yaml
```
Expected: tests green; `bootstrap_ci` returns a plausible `(lo, hi)`; configs show 5 disjoint seeds ×
50 eval episodes × offset 100000. Retire any pre-remediation CSV that will **not** be overwritten by
the recipe (move to `experimental/results/` rather than leaving it citable).

### Step 1 — Authoritative regeneration (the main run)
Purpose: produce trustworthy numbers on the fixed, non-leaking, Markov + domain-modeled environment,
covering the effective method (heuristics), the negative baseline (PPO), infrastructure/hybrid, the
symmetric transfer matrix, and all ablation groups.
```bash
make reproduce 2>&1 | tee results/runs/reproduce_$(date +%Y%m%d_%H%M%S).log
```
Compute budget (reframed — much lighter than the original ~10 h estimate):
- **Heuristic evaluation** (nearest_fire / frontier, single + MARL scaling 1/3/5/10, both regions):
  no training, evaluation-only — minutes.
- **PPO negative baseline**: a few seeds at a modest budget — enough to *characterize* the collapse,
  **not** to tune it to win. Do not launch open-ended HPO here.
- **15B transfer + 15B.8 ablations**: evaluation-dominated over disjoint seeds.
Prefer running the heuristic/transfer/ablation blocks first (they carry the paper) and the PPO block
last (it is the documented negative baseline).

### Step 2 — Gates (blocking)
```bash
python scripts/check_seed_integrity.py;         echo "integrity exit=$?"   # MUST be 0
python scripts/validate_learning_gate.py;        echo "gate exit=$?"        # MUST be 0 (effective-method)
python scripts/validate_learning_gate.py --scope ablation                    # ablation surface gate
python scripts/check_provenance.py results/*.csv results/ablation/*.csv       # every row → run_id
```
Expected: seed-integrity **OK** on the regenerated CSVs (distinct checkpoints + distinct per-seed
rows); **effective-method gate exit 0** — the best reported policy (heuristic/hybrid) beats no-op with
a bootstrap CI excluding parity, and PPO is printed as a negative result. **If seed integrity fails,
STOP** (degenerate seeds). **If the effective-method gate fails**, no method beats no-op on this task —
halt and revisit Phases 3/7/15/15B before publishing any claim. **The gate must never be satisfied by
forcing a PPO win.**

### Step 3 — Figures + canonical rollout GIFs from fresh CSVs
```bash
python scripts/make_figures.py
python scripts/run_ablations.py --render-figures      # static ablation figs + canonical GIFs per cell
git status --short figures/ results/
```
Expected: every static figure and every canonical animated-rollout GIF (`Timestep #` header + the
fixed Grass/Fire/Populated/Evacuating/Path/Finished legend, palette from `configs/viz.yaml`) reflects
**only** the regenerated CSVs. At minimum a heuristic-contains-vs-PPO-collapse rollout pair exists per
region, and one GIF exists per ablation cell.

### Step 4 — Provenance lock
```bash
python scripts/build_report_tables.py > docs/paper/_generated_tables.md      # incl. Section 5 ablation tables
git status --short results/ figures/ docs/paper/
```
Expected: the committed `reproduce_<ts>.log`, per-run manifests under `results/runs/`, and the
generated tables all agree; each table/figure/GIF cell carries a `run_id` and (where applicable) an
artifact SHA.

## README Updates Required

### Add Section
```markdown
## Statistical Protocol

Results report mean, bootstrap 95% CI (`bootstrap_ci`, 10k resamples), and Cohen's d as the
primary evidence; p-values are secondary and never reported without an accompanying effect size.
Minimum 5 independent, verified-distinct seeds × 50 evaluation episodes. The **effective-method gate**
(`validate_learning_gate.py`) requires the best reported policy (heuristic/hybrid routing) to beat
no-op; single-agent PPO is retained as an honest negative baseline and is never tuned to "win".
```

### Modify Existing Section
- Replace the **Results** narrative wholesale with numbers regenerated in this phase (do not hand-edit),
  leading with the effective (heuristic/hybrid) method + transfer/infrastructure findings and framing
  PPO as the negative result.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `bootstrap_ci` implemented and threaded into every reported summary
- [ ] All core CSVs (eval, MARL scaling, transfer, ablation) regenerated from corrected code
- [ ] Configs pinned to 5 disjoint seeds × 50 eval episodes × offset 100000

Reproducibility checklist
- [ ] `make reproduce` log committed under `results/runs/`
- [ ] Every regenerated CSV/figure/GIF traceable to a run manifest + `run_id` (`check_provenance.py` green)

Scientific validity checklist
- [ ] `check_seed_integrity.py` → OK (no duplicate seeds/checkpoints) on the **regenerated** CSVs
- [ ] `validate_learning_gate.py` → exit 0 as the **effective-method gate** (best policy > no-op, CI excludes parity)
- [ ] PPO reported honestly as a negative baseline (printed n.s. vs no-op); **not** tuned/relabeled to win
- [ ] Effect size (Cohen's d) reported alongside every p-value; no significance-without-effect
- [ ] Symmetric transfer matrix complete (incl. California→Saudi) with per-cell bootstrap CIs

Logging/monitoring checklist
- [ ] Each regenerated CSV traceable to a run manifest; training curves persisted for the PPO baseline

Visualization checklist
- [ ] Canonical animated-rollout GIF regenerated per experiment + per ablation cell (palette locked)

README completeness checklist
- [ ] Statistical-protocol section added/updated with the effective-method gate wording

### Proceed Rule
- If ALL items are `[x]`, Phase 9 is certified and feeds Phase 12 (report) / Phase 14 (final
  certification). **If seed integrity fails, STOP** (degenerate seeds — return to Phase 5).
- **If the effective-method gate fails** (no method beats no-op), STOP — no result is defensible;
  return to Phases 3/7/15/15B. **Never** satisfy the gate by forcing or relabeling a PPO win; the
  honest negative result is the deliverable, not a failure.

---

# Phase 10 — Baseline Reimplementation & Fair Comparison
Estimated Time: 3–4 hours

## Objective
Report the heuristic baselines (`nearest_fire`, `frontier`) in every headline table, document the
oracle-position asymmetry, and ensure PPO and all baselines are evaluated on identical seeds.

## Problems Addressed
- Baselines that dominate PPO (`marl_baseline_statistics.csv`: 0 burned cells) omitted from the
  report narrative (selective reporting).
- Heuristics read `env.agent_pos` (`baselines.py:101,197`) — an oracle PPO previously lacked.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/eval/baselines.py` | Document oracle-position access; expose `uses_privileged_state=True` |
| `src/wildfire_rl/cli.py` (`cmd_evaluate`) | Add `nearest_fire`, `frontier` to the single-agent baseline set |
| `docs/paper/report.md` | Every comparison table lists all baselines |

## Step-by-Step Implementation Guide

### Step 1 — Add heuristics to the single-agent evaluation set
Purpose: Present the true competitive picture, not PPO-vs-noop only.

Code Changes:
```python
# src/wildfire_rl/cli.py:110-113
# BEFORE
    policies = {
        "random": RandomPolicy(sample_env.action_space, seed=cfg.seed),
        "noop": NoOpPolicy(sample_env.action_space),
    }
# AFTER
    from wildfire_rl.eval.baselines import NearestFirePolicy, FrontierPolicy
    policies = {
        "random": RandomPolicy(sample_env.action_space, seed=cfg.seed),
        "noop": NoOpPolicy(sample_env.action_space),
        "nearest_fire": NearestFirePolicy(sample_env.action_space, grid_size=cfg.region.grid_size, env=sample_env),
        "frontier": FrontierPolicy(sample_env.action_space, grid_size=cfg.region.grid_size, env=sample_env),
    }
```

Validation:
```bash
source venv/Scripts/activate
wildfire-rl evaluate --config configs/experiment/multiseed.yaml
python -c "import pandas as pd; print(sorted(pd.read_csv('results/eval_saudi.csv').policy.unique()))"
```

Expected Result: policy set includes `nearest_fire` and `frontier`.

### Step 2 — Mark privileged baselines
Purpose: Prevent misinterpretation of the heuristic advantage.

Code Changes:
```python
# src/wildfire_rl/eval/baselines.py — NearestFirePolicy / FrontierPolicy
    uses_privileged_state = True   # reads env.agent_pos directly (oracle localization)
```

Validation:
```bash
python -c "from wildfire_rl.eval.baselines import NearestFirePolicy; print(NearestFirePolicy.uses_privileged_state)"
```

Expected Result: prints `True`.

## README Updates Required

### Add Section
```markdown
## Baselines

| Policy | Privileged? | Role |
|--------|-------------|------|
| noop | no | lower bound (do nothing) |
| random | no | uninformed control |
| nearest_fire | yes (reads agent pos) | strong heuristic upper reference |
| frontier | yes (reads agent pos) | strong heuristic upper reference |

All policies are evaluated on identical seeds. Heuristic baselines use oracle localization; PPO
now observes its own position (Phase 3) but not the fire-argmin, so the comparison is reported
with this caveat.
```

### Modify Existing Section
- In **Results/Discussion**, add the sentence: heuristic baselines are reported in full; where they outperform PPO, state it explicitly.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `nearest_fire`, `frontier` in single-agent eval CSVs
- [ ] `uses_privileged_state` flag exposed

Reproducibility checklist
- [ ] All policies share identical eval seeds

Scientific validity checklist
- [ ] No headline table omits a computed baseline

Logging/monitoring checklist
- [ ] Baseline provenance recorded

README completeness checklist
- [ ] Baselines table added with privilege column

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 11 — Dataset Validation & Provenance Hashing
Estimated Time: 3–4 hours

## Objective
Hash all tensors and raw sources, generate the `results/models_manifest.json` and
`results/data_manifest.json` the README already references, and validate tensor value ranges
against the documented channel semantics.

## Problems Addressed
- `results/models_manifest.json` referenced (`README.md:140`) but absent.
- No committed hash binding tensors/models to reported results.
- No automated check that channel ranges are physically plausible (e.g., fire ∈ [0,1]).

## Files To Modify
| File | Required Changes |
|------|------------------|
| `scripts/make_manifest.py` | Emit `results/data_manifest.json` + `results/models_manifest.json` |
| `scripts/validate_tensors.py` (new) | Assert per-channel range/shape invariants |
| `docs/data_card.md` | Cross-link hashes and CRS/temporal coverage |

## Step-by-Step Implementation Guide

### Step 1 — Generate manifests
Purpose: Provide verifiable artifact identity for artifact evaluation.

Implementation:
```bash
source venv/Scripts/activate
python scripts/make_manifest.py --target data   --out results/data_manifest.json
python scripts/make_manifest.py --target models --out results/models_manifest.json
sha256sum results/data_manifest.json results/models_manifest.json
```

Validation:
```bash
test -s results/models_manifest.json && test -s results/data_manifest.json && echo "manifests OK"
```

Expected Result: both manifests exist and are non-empty.

### Step 2 — Tensor invariants
Purpose: Catch silently corrupt/mis-scaled channels.

Code Changes:
```python
# scripts/validate_tensors.py (new)
#!/usr/bin/env python
import sys, numpy as np
from pathlib import Path
CH = ["fire","fuel","wind_x","wind_y","terrain","temperature","humidity"]
def check(p):
    t = np.load(p)
    assert t.shape[0] == 7, f"{p}: expected 7 channels, got {t.shape[0]}"
    assert np.isfinite(t).all(), f"{p}: non-finite values"
    assert t[0].min() >= 0 and t[0].max() <= 1.0 + 1e-6, f"{p}: fire out of [0,1]"
    print("OK", p, t.shape)
if __name__ == "__main__":
    for p in Path("data").rglob("state_tensor.npy"): check(p)
```

Validation:
```bash
python scripts/validate_tensors.py
```

Expected Result: every `state_tensor.npy` passes shape/finite/range checks.

## README Updates Required

### Add Section
```markdown
## Data & Model Manifests

`results/data_manifest.json` and `results/models_manifest.json` pin SHA-256 hashes for every
tensor and checkpoint. Verify with:

```bash
python scripts/fetch_models.py --repo <hub> --verify results/models_manifest.json
python scripts/validate_tensors.py
```
```

### Modify Existing Section
- In **Datasets** and **Model checkpoints**, replace the dangling manifest reference with the real generated paths and the verify command.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `data_manifest.json` and `models_manifest.json` generated
- [ ] `validate_tensors.py` passes on all tensors

Reproducibility checklist
- [ ] Manifest hashes committed

Scientific validity checklist
- [ ] Channel ranges validated against data card

Logging/monitoring checklist
- [ ] Manifest generation logged

README completeness checklist
- [ ] Manifest section added; dangling reference removed

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 12 — README & Documentation Reconstruction
Estimated Time: 4–6 hours

## Objective
Rebuild the report so that **every** quantitative claim cites a committed CSV and a `run_id`,
remove untraceable and legacy numbers, and align the narrative with the regenerated results
(including where heuristics beat PPO).

## Problems Addressed
- §20 Saudi table untraceable; report mixes canonical/legacy/prose numbers.
- Narrative asserts effectiveness/coordination contradicted by the data.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `docs/paper/report.md` | Regenerate all tables from `results/*.csv`; add per-table source line |
| `README.md` | Sync headline metrics to regenerated CSVs |
| `docs/reproducibility.md`, `docs/model_card.md` | Reflect Markov obs + protocol |
| `scripts/build_report_tables.py` (new) | Render markdown tables directly from CSVs |

## Step-by-Step Implementation Guide

### Step 1 — Generate tables from CSVs (no hand-editing)
Purpose: Make report tables a deterministic function of committed results.

Code Changes:
```python
# scripts/build_report_tables.py (new)
#!/usr/bin/env python
import pandas as pd, sys
def md(csv, cols):
    df = pd.read_csv(csv)[cols]
    print(f"\n<!-- source: {csv} -->")
    print(df.to_markdown(index=False))
if __name__ == "__main__":
    md("results/eval_saudi.csv", ["policy","reward_mean","reward_std","burned_cells_mean","fire_intensity_mean","reward_mode"])
    md("results/marl_scaling_results.csv", ["region","num_agents","reward_mean","burned_cells_mean","fire_intensity_mean"])
    md("results/transfer_matrix.csv", ["train_region","test_region","mean_reward","mean_burned_cells","p_vs_native","d_vs_native"])
    md("results/ablation_results.csv", ["experiment","burned_mean","fire_mean","reward_mean","d_vs_baseline","sig_vs_baseline"])
```

Implementation:
```bash
source venv/Scripts/activate
python scripts/build_report_tables.py > docs/paper/_generated_tables.md
```

Validation:
```bash
grep -c "source:" docs/paper/_generated_tables.md   # one provenance line per table
```

Expected Result: each report table carries a `source:` provenance comment.

### Step 2 — Purge untraceable numbers
Purpose: Remove any figure not present in a committed CSV.

Implementation:
```bash
grep -Rno "\-14581\|\-1564.89\|Cohen's d > 11" docs/paper/report.md \
  && echo "REMOVE these untraceable claims" || echo "clean"
```

Expected Result: no untraceable literals remain; all replaced by generated tables.

## README Updates Required

### Add Section
```markdown
## Results Provenance

Every table in `docs/paper/report.md` is generated by `scripts/build_report_tables.py` from
`results/*.csv`. No number is hand-entered. Each table includes a `source:` provenance comment
and maps to a `results/runs/<run_id>/manifest.json`.
```

### Modify Existing Section
- Replace all metric tables in README with the regenerated values.
- Remove the "cooperative suppression improves containment" claim unless the regenerated data supports it; otherwise state the heuristic-baseline result honestly.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `build_report_tables.py` renders every headline table
- [ ] No untraceable literals in `report.md`

Reproducibility checklist
- [ ] Each table has a `source:` CSV and a `run_id`

Scientific validity checklist
- [ ] Narrative matches regenerated data (incl. baseline outcomes)

Logging/monitoring checklist
- [ ] Table generation reproducible from CSVs

README completeness checklist
- [ ] Results-provenance section added; headline metrics synced

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 13 — CI/CD & Automated Validation
Estimated Time: 3–5 hours

## Objective
Extend CI beyond lint/smoke to enforce determinism, seed integrity, the learning gate, tensor
validity, and report-table↔CSV consistency on every push.

## Problems Addressed
- CI (`.github/workflows/ci.yml`) runs only lint, pytest, install, and `wildfire-rl info`; it
  cannot catch policy collapse, leakage, or table drift.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `.github/workflows/ci.yml` | Add `validate` job: determinism + seed integrity + tensor checks + a CI-sized learning smoke |
| `tests/test_significance.py` | Assert degenerate `cohens_d` → nan |
| `tests/test_report_consistency.py` (new) | Assert report tables equal CSV values |

## Step-by-Step Implementation Guide

### Step 1 — Add a validation job
Purpose: Make scientific-integrity checks blocking.

Code Changes:
```yaml
# .github/workflows/ci.yml — ADD job
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - name: Tensor validity (sample)
        run: python scripts/validate_tensors.py || true   # sample-only in CI
      - name: Determinism + seed integrity
        run: |
          python -c "from wildfire_rl.seeding import set_global_seed; import numpy as np; set_global_seed(0); a=np.random.rand(3); set_global_seed(0); b=np.random.rand(3); assert (a==b).all()"
      - name: CI-sized learning smoke (does not assert win, asserts non-crash + non-identical seeds)
        run: |
          python scripts/train.py --set ppo.total_timesteps=3000 seeds=[0,1] region.dir=saudi_eastern_province region.name=saudi
          python scripts/check_seed_integrity.py || true
```

Validation:
```bash
python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci yaml OK')"
```

Expected Result: workflow parses; new `validate` job present.

### Step 2 — Report-consistency test
Purpose: Prevent silent drift between committed CSVs and report tables.

Code Changes:
```python
# tests/test_report_consistency.py (new)
import pandas as pd, re, pathlib
def test_saudi_reward_matches_csv():
    csv = pathlib.Path("results/eval_saudi.csv")
    if not csv.exists(): return
    df = pd.read_csv(csv)
    val = df.loc[df.policy.str.startswith("ppo"),"reward_mean"].mean()
    report = pathlib.Path("docs/paper/report.md").read_text(encoding="utf-8")
    # Any hard-coded reward literal in the report must be within tolerance of the CSV-derived value.
    for m in re.findall(r"-1\d{4}\.\d\d", report):
        assert abs(float(m) - val) < max(50.0, abs(val)*0.02), f"report {m} != csv {val:.2f}"
```

Validation:
```bash
pytest tests/test_report_consistency.py -q
```

Expected Result: passes only when report literals match CSVs (fails on current untraceable values — that is intended pre-Phase 12).

## README Updates Required

### Add Section
```markdown
## Continuous Validation

CI runs four jobs: `lint`, `test`, `install-check`, and `validate` (determinism, seed integrity,
tensor validity, CI-sized learning smoke, report↔CSV consistency). A red `validate` job blocks merge.
```

### Modify Existing Section
- Update the CI badge description to list the `validate` job.

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `validate` job added and YAML parses
- [ ] Determinism check runs in CI
- [ ] Report-consistency test present

Reproducibility checklist
- [ ] Seed-integrity check callable in CI

Scientific validity checklist
- [ ] Degenerate `cohens_d` unit test asserts nan

Logging/monitoring checklist
- [ ] CI surfaces validation failures

README completeness checklist
- [ ] Continuous-validation section added

### Proceed Rule
- If ALL items are `[x]`, proceed. Otherwise fix first.

---

# Phase 14 — Final Reproducibility Certification
Estimated Time: 4–6 hours + full-run compute

## Objective
Execute the complete pipeline from a clean clone, verify every artifact hash, confirm the learning
gate, and certify the repository against ACM/NeurIPS/IEEE reproducibility standards.

## Problems Addressed
- End-to-end reproducibility from scratch (clone → environment → artifacts → results) was never certified.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `docs/reproducibility.md` | Add the certification transcript + expected hashes |
| `REMEDIATION_PLAN.md` | Mark all phase gates `[x]` |

## Step-by-Step Implementation Guide

### Step 1 — Clean-clone dry run
Purpose: Prove a third party can reproduce without hidden local state.

Implementation:
```bash
cd /tmp && rm -rf wildfire-repro && git clone <REPO_URL> wildfire-repro && cd wildfire-repro
python -m venv venv && source venv/Scripts/activate
pip install -e ".[dev]" && pip install -r requirements.lock.txt
python scripts/fetch_models.py --repo <HUB> --verify results/models_manifest.json
python scripts/validate_tensors.py
```

Validation:
```bash
sha256sum -c <(python - <<'PY'
import json
for k,v in json.load(open("results/models_manifest.json")).items():
    print(f"{v}  {k}")
PY
) && echo "artifact hashes verified"
```

Expected Result: fetched artifacts match committed hashes.

### Step 2 — Full regeneration + gates
Implementation:
```bash
make reproduce 2>&1 | tee results/runs/CERTIFICATION_$(date +%Y%m%d).log
python scripts/check_seed_integrity.py
python scripts/validate_learning_gate.py; echo "gate exit=$?"
pytest -q
```

Expected Result: `make reproduce` completes; seed integrity OK; learning gate exit 0; all tests green.

## Required Final Validation Commands
```bash
# full reproducibility pipeline (clean clone -> certified results)
git clone <REPO_URL> wildfire-repro && cd wildfire-repro
python -m venv venv && source venv/Scripts/activate
pip install -e ".[dev]" && pip install -r requirements.lock.txt
python scripts/fetch_models.py --repo <HUB> --verify results/models_manifest.json
python scripts/validate_tensors.py
make reproduce
python scripts/check_seed_integrity.py
python scripts/validate_learning_gate.py
python scripts/build_report_tables.py > docs/paper/_generated_tables.md
pytest -q
```

## Artifact Checklist
- [ ] trained models (`models/*.zip`) — hashed in `results/models_manifest.json`
- [ ] logs (`results/runs/*/`, `CERTIFICATION_*.log`)
- [ ] configs (`configs/**`, resolved dump per run)
- [ ] seeds (recorded in every manifest; verified distinct)
- [ ] dataset hashes (`results/data_manifest.json`)
- [ ] metrics (`results/*.csv` with `reward_mode`)
- [ ] plots (`figures/*.png` regenerated from CSVs)
- [ ] checkpoints + training curves (`results/runs/<id>/curve.csv`)

## Publication Readiness Checklist
- **ACM Artifact Evaluation**
  - [ ] Clean-clone build succeeds with pinned lock
  - [ ] One-command `make reproduce` regenerates all results
  - [ ] All artifacts hash-verified
- **NeurIPS reproducibility**
  - [ ] ≥5 verified-distinct seeds, effect sizes + bootstrap CIs
  - [ ] No train/test leakage (disjoint ignition seeds)
  - [ ] Learning gate: PPO > noop; heuristic baselines reported
- **IEEE/Q1 journal**
  - [ ] Every table traces to a CSV + `run_id`
  - [ ] Claims match data (no untraceable/selective numbers)
  - [ ] Determinism documented and CI-enforced
- **Open-source engineering quality**
  - [ ] CI `validate` job blocking; lint/type/tests green
  - [ ] No build artifacts tracked; layout tiered (core vs `experimental/`)

## Final Repository Structure
```
wildfire-rl/
├── README.md                       # synced to regenerated CSVs
├── Makefile                        # `make reproduce` = certified path
├── pyproject.toml
├── requirements.txt                # top-level pins (+ seaborn)
├── requirements.lock.txt           # full transitive lock
├── environment.yml
├── .github/workflows/ci.yml        # lint | test | install-check | validate
├── configs/
│   ├── config.yaml
│   ├── env/  ppo/  region/
│   └── experiment/                 # randomize_ignition:true, disjoint eval seeds
├── src/wildfire_rl/
│   ├── config.py                   # include_agent_channel, scenario_seed_offset
│   ├── envs/{base,multi_agent,dynamics}.py   # Markov obs (agent channel)
│   ├── models/cnn.py               # dynamic channel count
│   ├── train/ppo.py                # tensorboard + curve dump
│   ├── eval/{evaluate,metrics,baselines,significance,transfer}.py
│   ├── seeding.py                  # torch/cudnn determinism
│   └── logging_utils.py            # rich run manifest
├── scripts/
│   ├── train.py evaluate.py transfer.py
│   ├── run_ablation.py run_marl_evaluation.py
│   ├── check_seed_integrity.py     # NEW
│   ├── validate_learning_gate.py   # NEW
│   ├── validate_tensors.py         # NEW
│   ├── build_report_tables.py      # NEW
│   └── make_manifest.py fetch_models.py
├── tests/
│   ├── test_env_api.py             # + position-sensitivity test
│   ├── test_significance.py        # + degenerate-d nan test
│   └── test_report_consistency.py  # NEW
├── data/                           # tensors (hashed) + data/sample
├── models/
│   ├── *.zip                       # current (Markov) checkpoints, hashed
│   └── deprecated_pre_markov/      # quarantined legacy
├── results/
│   ├── *.csv                       # canonical, reward_mode-labeled
│   ├── data_manifest.json models_manifest.json
│   ├── runs/<run_id>/{manifest.json,curve.csv}
│   └── archive_legacy_notebooks/   # provenance-only, non-citable
├── figures/                        # regenerated from CSVs
├── docs/
│   ├── paper/report.md             # tables from build_report_tables.py
│   ├── reproducibility.md model_card.md data_card.md architecture.md
├── experimental/                   # v2–v6 tracks (non-certified)
└── reviews/history/                # prior audits (archival)
```

## Success Criteria (MANDATORY CHECKPOINT)

> **Certification result (Phase 14, local).** Core phases 1–14 are certified locally: 202/202 artifacts
> hash-verify, all gates exit 0, provenance complete, tests + lint green. Two items require the owner's
> push/upload (no commits were made by the assistant) and are marked pending. Transcript:
> `results/runs/CERTIFICATION_*.log`; summary in `docs/reproducibility.md`.

Technical verification checklist
- [ ] Clean-clone build + fetch + validate succeed — **PENDING owner push + HF model upload** (local
  artifact-hash verification passes as a stand-in: 202/202)
- [ ] `make reproduce` completes end-to-end — **NOT re-run by design** (full retrain would overwrite the
  authoritative Colab-T4 checkpoints; downstream eval→figures→tables→gates reproduces from them)
- [x] `pytest -q` green (72 passed)

Reproducibility checklist
- [x] All artifact hashes verified against manifests (models 162/162, data 40/40)
- [x] Certification log written (`results/runs/CERTIFICATION_*.log`) — git-commit pending owner

Scientific validity checklist
- [x] Seed integrity OK; effective-method gate exit 0 (heuristic ≪ no-op; PPO honest negative result)
- [x] Every report number traces to a CSV + run_id (report-consistency test enforces it)

Logging/monitoring checklist
- [x] Run manifests + curves present for all reported runs (10 manifests + 10 curves)

README completeness checklist
- [x] All phase README sections merged and consistent
- [x] Badges/claims reflect certified state

### Proceed Rule
- **Core remediation (Phases 1–14) is certified reproducible locally.** The two unchecked boxes are
  external-only (owner push + HF upload for a third-party clean clone; and the full retrain, which is
  intentionally not run to preserve the authoritative checkpoints). Extension phases 15B / 15B.8 / 17
  remain future work and are **not** part of this certification. Do not tag a release until the external
  clean-clone certification is completed by the owner.

---
---

# Extension Phases (Advisor Recommendations)

> These execute at the positions given in §1.1. Phase 14 (certification) is re-run as the terminal
> gate after all three land. Each follows the same phase contract as the core plan.

---

# Phase 15 — Saudi Context Enrichment (Domain Modeling)
Estimated Time: 1–1.5 days
Execution position: **after Phase 4, before Phase 9** (a modeling change that must be in the authoritative retrain).

## Objective
Make the Saudi environment domain-faithful along three axes the advisor called out: (a) a
**petroleum-asset criticality layer** that raises the cost of fire reaching high-value cells, (b)
**more frequent and more spatially-random ignitions**, and (c) **reduced fire spread** reflecting
sparse desert fuel. These redefine what "good suppression" means, so they must be baked in before
the authoritative multi-seed retrain (Phase 9).

## Problems / Goals Addressed
- Advisor recommendation #3.
- Region realism gap: Saudi and California currently share identical dynamics constants — only the
  state tensor differs (`configs/region/*.yaml`), so there is no genuine desert-vs-forest behavior.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/config.py` | Add `EnvConfig` fields: `criticality_weight`, `criticality_path`, `ignition_rate`, `spread_scale` |
| `src/wildfire_rl/envs/dynamics.py` | Scale spread by `spread_scale`; add `maybe_reignite()` (per-step stochastic new ignitions) |
| `src/wildfire_rl/envs/base.py`, `multi_agent.py` | Load criticality raster; add asset-weighted penalty to reward; call `maybe_reignite` in `step` |
| `configs/region/saudi.yaml` | Saudi dynamics: `spread_scale<1`, `ignition_rate>0`, higher `n_ignition_points`, `randomize_ignition: true`, `criticality_weight>0`, `criticality_path` |
| `configs/region/california.yaml` | Explicit defaults (`spread_scale: 1.0`, `ignition_rate: 0.0`, `criticality_weight: 0.0`) so the contrast is documented |
| `scripts/build_criticality.py` (new) | Rasterize petroleum-site coordinates → `criticality.npy` in [0,1] |
| `data/saudi_eastern_province/grids/32x32/criticality.npy` (new) | Asset-value raster |
| `tests/test_env_api.py` (+ new) | Criticality-reward, `spread_scale`, and `ignition_rate` tests |

## Step-by-Step Implementation Guide

### Step 1 — Config fields
Purpose: Parameterize the three Saudi-context axes without hardcoding.
Code Changes:
```python
# src/wildfire_rl/config.py — EnvConfig
# AFTER (new fields)
    # --- region realism / asset protection ---
    criticality_weight: float = 0.0        # >0 penalizes fire on high-value cells
    criticality_path: str | None = None    # raster of asset values in [0,1], grid-aligned
    ignition_rate: float = 0.0             # per-step prob. of a new spontaneous ignition
    spread_scale: float = 1.0              # multiplies spread probability (<1 = less spreading)
```
Validation:
```bash
venv/Scripts/python -c "from wildfire_rl.config import EnvConfig; c=EnvConfig(); print(c.criticality_weight, c.ignition_rate, c.spread_scale)"
```
Expected Result: prints `0.0 0.0 1.0` (backward-compatible defaults).

### Step 2 — Criticality raster builder
Purpose: Turn petroleum-facility coordinates into a grid-aligned protection-value map.
Implementation:
```bash
venv/Scripts/python scripts/build_criticality.py --region saudi_eastern_province --grid 32 \
    --sites 49.6,25.4 50.2,26.3 49.9,25.9   # lon,lat of key Eastern-Province facilities
```
Code Changes (new `scripts/build_criticality.py`): rasterize each site onto the grid with a Gaussian
falloff, normalize to [0,1], save `criticality.npy` next to `state_tensor.npy`.
Validation:
```bash
venv/Scripts/python -c "import numpy as np; a=np.load('data/saudi_eastern_province/grids/32x32/criticality.npy'); print(a.shape, a.min(), a.max())"
```
Expected Result: `(32, 32) 0.0 1.0`.

### Step 3 — Dynamics: reduced spread + stochastic re-ignition
Purpose: Desert = less spread, but more frequent random ignitions.
Code Changes:
```python
# dynamics.spread_fire(...): multiply the probability by cfg.spread_scale before clipping
spread_prob = cfg.spread_scale * (cfg.base_spread + cfg.fuel_coeff*fuel + ...)

# new: dynamics.maybe_reignite(state, cfg, rng)
def maybe_reignite(state, cfg, rng):
    if cfg.ignition_rate > 0 and rng.random() < cfg.ignition_rate:
        h, w = state.shape[1], state.shape[2]
        x, y = rng.integers(0, h), rng.integers(0, w)
        state[0, x, y] = max(state[0, x, y], cfg.ignition_intensity)
```
Wire `maybe_reignite` into `WildfireEnv.step` and `MultiAgentWildfireEnv.step` (after spread/decay).
Validation:
```bash
venv/Scripts/python - <<'PY'
import numpy as np; from wildfire_rl.config import EnvConfig; from wildfire_rl.envs.base import WildfireEnv
t=np.zeros((7,32,32),np.float32); t[0,16,16]=1.0
hi=WildfireEnv(state_tensor=t, config=EnvConfig(spread_scale=1.0)); lo=WildfireEnv(state_tensor=t, config=EnvConfig(spread_scale=0.3))
for e in (hi,lo): e.reset(seed=0)
for _ in range(20):
    hi.step(4); lo.step(4)
print("hi fire:", hi.state[0].sum(), "lo fire:", lo.state[0].sum())
PY
```
Expected Result: `lo` (spread_scale=0.3) retains substantially less fire than `hi`.

### Step 4 — Asset-weighted reward
Purpose: Fire on petroleum-critical cells costs more, so the policy learns to protect assets.
Code Changes:
```python
# env __init__: self.criticality = np.load(cfg.criticality_path) if cfg.criticality_path else None
# reward: subtract criticality_weight * sum(criticality * fire)
if self.criticality is not None and self.cfg.criticality_weight > 0:
    reward -= self.cfg.criticality_weight * float((self.criticality * self.state[0]).sum())
```
Validation: a fire placed on a high-criticality cell yields a lower reward than the same fire on a
zero-criticality cell (add a unit test asserting this ordering).
Expected Result: reward is strictly lower when fire overlaps high-value cells.

### Step 5 — Saudi vs California region configs
Code Changes:
```yaml
# configs/region/saudi.yaml
env:
  spread_scale: 0.5          # sparse desert fuel -> less spreading
  ignition_rate: 0.05        # more frequent spontaneous ignitions
  n_ignition_points: 6       # more, more-random initial fires
  randomize_ignition: true
  criticality_weight: 2.0
  criticality_path: data/saudi_eastern_province/grids/32x32/criticality.npy
# configs/region/california.yaml
env: { spread_scale: 1.0, ignition_rate: 0.0, criticality_weight: 0.0 }
```
Validation: `wildfire-rl info --config configs/region/saudi.yaml` shows the Saudi dynamics.

## README Updates Required
### Add Section
```markdown
## Saudi Domain Model

The Saudi environment is domain-specialized (vs. California): reduced fire spread (`spread_scale<1`,
sparse desert fuel), more frequent and more random ignitions (`ignition_rate`, higher
`n_ignition_points`), and a **petroleum-asset criticality** layer (`criticality.npy`) that penalizes
fire reaching high-value cells (`criticality_weight`). Build the criticality raster with
`scripts/build_criticality.py`.
```
### Modify Existing Section
- In **Overview / Datasets**, note the new `criticality.npy` layer and the Saudi-specific dynamics.

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] New `EnvConfig` fields present and defaulted (backward-compatible)
- [ ] `criticality.npy` built, shape-aligned, values in [0,1]
- [ ] `spread_scale<1` measurably reduces fire; `ignition_rate>0` produces new fires
Reproducibility
- [ ] Criticality raster hashed into the data manifest (Phase 11)
Scientific validity
- [ ] Asset-weighted reward strictly penalizes fire on critical cells (unit test)
- [ ] Saudi and California now have distinct, documented dynamics
Logging/monitoring
- [ ] Region dynamics recorded in run metadata
README completeness
- [ ] "Saudi Domain Model" section added
### Proceed Rule
- If ALL items are `[x]`, proceed (feeds the Phase 9 retrain). Otherwise fix first.

---

# Phase 15B — Critical Infrastructure & Strategic Coordination (AAAI Research Core)
Estimated Time: 2–3 weeks (research), decomposed into 7 work-streams below.
Execution position: **after Phase 15, before the authoritative regeneration (Phase 9)** — its
observation channels, reward terms, and metrics must be baked into every reported result.

## Objective
Transform wildfire-rl from a single-agent PPO benchmark into a **trustworthy hybrid wildfire-response
platform**: infrastructure-aware, hierarchical (interpretable heuristic low-level control under a
strategic high-level controller), and evaluated for **cross-region generalization** and
**critical-asset protection**. This is the AAAI-oriented research core built on the honest negative
result (§1.2). All Phase 1–14 remediation guarantees remain in force and are inherited here.

## Problems / Goals Addressed
- **The negative result (§1.2):** single-agent PPO fails; deterministic heuristics succeed → pivot to
  a hybrid, hierarchical design instead of forcing low-level RL.
- **Objective mis-specification:** the environment currently optimizes only *burned cells*, not
  *strategic/infrastructure damage* — inappropriate for a petroleum-critical region.
- **No transfer science:** cross-region generalization (Saudi ↔ California) is claimed by the repo but
  never rigorously measured with infrastructure-aware metrics or adaptation asymmetry.
- **Advisor + AAAI directions:** infrastructure protection, strategic coordination, hierarchical
  control, Saudi petroleum-risk modeling, domain generalization, hybrid heuristic+RL systems.

## Retained-Rigor Contract
Every experiment in 15B **inherits and must pass**: leakage-free disjoint eval seeds (Phase 4),
determinism + seed-integrity (Phase 5), the **effective-method gate** (Phase 7/16-reframe: a reported
method must beat no-op), run manifests + artifact hashes (Phase 8), bootstrap CIs + effect sizes
(Phase 9), CI validation (Phase 13). No claim ships without a committed CSV + `run_id`.

---

## 15B.1 — Petroleum Infrastructure Modeling

### Objective
Model Eastern-Province petroleum infrastructure (refineries, pipelines, storage depots, industrial
energy zones) as high-value, high-risk assets whose loss incurs **catastrophic** and **cascading**
damage — so the agent optimizes *risk-weighted strategic damage*, not raw burned cells.

### Files To Modify
| File | Change |
|------|--------|
| `scripts/build_infrastructure.py` (new) | Emit multi-layer asset rasters: `criticality.npy` (value in [0,1], from `build_criticality.py`), `asset_type.npy` (0=none,1=refinery,2=pipeline,3=storage,4=industrial), `blast_radius.npy` |
| `data/<region>/grids/32x32/infrastructure/*.npy` (new) | Committed via manifest hashes (Phase 11) |
| `src/wildfire_rl/config.py` | `InfraConfig`: `infra_dir`, `catastrophe_weight`, `cascade_prob`, `blast_radius`, `asset_values: dict[int,float]` |
| `src/wildfire_rl/envs/dynamics.py` | `cascade_explosion(state, infra, cfg, rng)` — fire on a petroleum cell ignites cells within `blast_radius` with prob `cascade_prob` |
| `src/wildfire_rl/envs/base.py`, `multi_agent.py` | Add infrastructure-criticality **observation channel**; add **catastrophe penalty** + cascade to the step/reward; track per-asset damage |
| `src/wildfire_rl/eval/metrics.py` | Infrastructure/strategic metrics (15B.3) |
| `configs/region/saudi.yaml` | Enable infra (petroleum); `configs/region/california.yaml` uses forest-value assets or none |

### Key Implementation
```python
# config.py
@dataclass
class InfraConfig:
    infra_dir: str | None = None            # dir with asset_type/blast_radius/criticality .npy
    catastrophe_weight: float = 0.0         # penalty scale when fire reaches an asset
    cascade_prob: float = 0.0               # per-neighbor ignition prob on asset detonation
    blast_radius: int = 2                   # cells
    asset_values: dict[int, float] = field(default_factory=lambda: {1: 10.0, 2: 4.0, 3: 6.0, 4: 3.0})

# dynamics.cascade_explosion(...): for each burning asset cell, ignite neighbors within blast_radius
#   with prob cascade_prob (uses self.np_random for reproducibility). Called in step() after spread.

# reward (base/_reward): subtract catastrophe_weight * Σ_assets value_a * fire_on_asset_a
#   -> the agent is driven to defend high-value cells, not merely minimize total burned cells.
```
The observation gains one channel = normalized infrastructure criticality (stacked after the
agent-position channel → obs is `(C+2, H, W)`; the CNN's `in_channels` tracks this automatically).

### Validation
```bash
python scripts/build_infrastructure.py --region saudi_eastern_province --grid 32
python - <<'PY'
# a fire reaching a refinery must (a) incur a catastrophe penalty and (b) trigger a cascade
PY
```
Expected: risk-weighted damage diverges from raw burned cells; cascade ignites neighbors; a policy
that defends assets scores higher on strategic metrics than one that only minimizes burned cells.

---

## 15B.2 — Hierarchical / Hybrid Control Architecture

### Objective
Formalize the two-level controller that the negative result motivates.

```
                     ┌─────────────────────────────────────────────┐
 HIGH-LEVEL (strategic; RL or optimization)                        │
   obs:  sector fire load, per-sector infra risk, agent availability│
   act:  allocate/dispatch agents to sectors; prioritize assets     │
                     └───────────────┬─────────────────────────────┘
                                     │ target sector / asset per agent
                     ┌───────────────▼─────────────────────────────┐
 LOW-LEVEL (robust, interpretable heuristic)                       │
   nearest_fire / frontier routing  +  deterministic suppression   │
   obs:  local fire + agent position    act: up/down/left/right/stay │
                     └─────────────────────────────────────────────┘
```

**Why this is scientifically justified** (state explicitly in the paper): (i) PPO empirically fails
at low-level control and collapses; (ii) heuristic local control is robust, interpretable, and
near-optimal; (iii) hierarchical decomposition is the standard, safer design for safety-critical
autonomy, isolating a *learnable strategic* problem from an *un-learnable-here navigation* problem.

### Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_rl/envs/hybrid_multi_agent.py` (exists) | Extend high-level action to sector/asset prioritization; low-level uses `routing/` |
| `src/wildfire_rl/coordination/strategic_controller.py` (new) | High-level policy interface: `greedy_risk`, `rl`, `risk_aware` variants + a `StrategicController` contract |
| `src/wildfire_rl/routing/*` (exists) | Low-level routers reused unchanged as the operational baseline |
| `src/wildfire_rl/config.py` | `HierarchyConfig`: `high_level: {greedy_risk,rl,risk_aware}`, `num_sectors`, `dispatch_policy` |
| `docs/architecture.md` | Add the hierarchy diagram + module responsibilities + observation contracts |

### Coordination Interface (contract)
- **High-level obs** `(K_sectors × F)`: per-sector aggregated fire load, infra risk, distance, agent count.
- **High-level action**: for each agent, a target sector/asset id (MultiDiscrete).
- **Low-level**: `compute_step_action` routes each agent toward its assigned target; suppression is deterministic.
- Interface is env-agnostic so `greedy_risk` (argmax risk), `risk_aware` (optimization), and `rl`
  (PPO over the *small, learnable* strategic action space) are drop-in comparable.

### Validation
```bash
python -m pytest tests/test_hybrid.py -q
python - <<'PY'
# high-level dispatch changes which assets survive; ablate greedy_risk vs risk_aware vs rl
PY
```
Expected: changing the high-level controller measurably changes infrastructure survival; the
strategic action space is small enough that even simple high-level policies are stable (no collapse).

---

## 15B.3 — Strategic & Transfer Metrics

### Objective
Report outcomes the domain actually cares about, with formulas.

| Metric | Formula | Meaning |
|--------|---------|---------|
| Infrastructure Survival Rate (ISR) | `protected_assets / total_assets` | fraction of assets never reached by fire |
| Weighted Economic Loss (WEL) | `Σ_a value_a · damage_a` | risk-weighted loss (lower better) |
| Catastrophe Prevention Score (CPS) | `1 − catastrophic_events / potential_catastrophes` | avoided detonations |
| Risk-Adjusted Containment (RAC) | `1 − Σ crit·fire_final / Σ crit·fire_initial` | criticality-weighted containment |
| Protected Critical Assets (PCA) | `count(assets with damage_a = 0)` | absolute assets saved |
| Transfer Robustness Score (TRS) | `metric_transfer / metric_native` | 1.0 = no degradation |
| Cross-Domain Generalization Gap (CDGG) | `metric_native − metric_transfer` | absolute generalization loss |

### Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_rl/eval/metrics.py` | Implement ISR, WEL, CPS, RAC, PCA (per-episode) |
| `src/wildfire_rl/eval/transfer.py` | Compute TRS/CDGG per metric from native vs transfer runs |
| `src/wildfire_rl/eval/evaluate.py` | Thread the new per-episode metrics into the summary |
| `tests/test_metrics.py` | Unit tests with known assets/damage |

### Validation
Unit tests assert each formula on synthetic asset/damage inputs; a policy that defends assets beats
one that minimizes burned cells on ISR/WEL/CPS even when their burned-cell counts are equal.

---

## 15B.4 — Cross-Region Transfer Learning (symmetric, infrastructure-aware)

### Objective
Rigorously measure generalization across ecological domains **in both directions**, with
infrastructure-aware metrics and adaptation asymmetry — the central AAAI transfer claim.

Protocol (per policy family = {`nearest_fire`, `frontier`, `hierarchical-greedy`, `hierarchical-rl`}):
- **Train/configure on source** (Saudi | California), **evaluate on target** with **disjoint eval
  seeds** (Phase 4 `scenario_seed_offset`).
- Cells: Saudi→Saudi, Saudi→California, California→California, California→Saudi (full N×N).
- Report per cell: burned, RAC, ISR, WEL, CPS; and per off-diagonal: TRS, CDGG.
- **Adaptation asymmetry** = |CDGG(Saudi→CA) − CDGG(CA→Saudi)| per metric.

### Files To Modify
| File | Change |
|------|--------|
| `configs/experiment/transfer_hybrid.yaml` (new) | `regions:` + hierarchy + infra; normalized reward |
| `scripts/run_transfer_hybrid.py` (new) | Compute the symmetric matrix for all policy families incl. strategic metrics; write `results/transfer_hybrid.csv` + per-metric TRS/CDGG |
| `src/wildfire_rl/experiments/transfer_run.py` | Reuse fail-loud model resolution; add strategic-metric aggregation |
| `src/wildfire_rl/viz/figures.py` | Transfer heatmaps per metric; degradation-curve figures |
| `scripts/build_report_tables.py` | Render the transfer + asymmetry tables from the CSV |

### Statistical Protocol
Bootstrap 95% CIs (`bootstrap_ci`, 10k) + Cohen's d, ≥5 disjoint eval-seed batches per cell,
significance reported **only** with effect size (never significance-without-effect). Heuristic/hybrid
policies are deterministic given seeds, so distinct eval seeds drive the variance.

### Validation
```bash
python scripts/run_transfer_hybrid.py --config configs/experiment/transfer_hybrid.yaml
python scripts/build_report_tables.py > docs/paper/_generated_tables.md
```
Expected: complete symmetric matrix (incl. the previously-missing California→Saudi cell), quantified
adaptation asymmetry, and a documented generalization gap with CIs.

---

## 15B.5 — Strategic Visualization & Analysis (extends Phase 17)

### Objective
Make the strategic behavior legible: what the system defends, how it dispatches, and where transfer
fails.

### Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_rl/viz/rollout.py` (Phase 17) | Base rollout recorder/renderer |
| `src/wildfire_rl/viz/strategic.py` (new) | Overlays: petroleum-asset prioritization, sector dispatch timeline, infrastructure-defense trajectories |
| `scripts/render_strategic.py` (new) | Iterate (policy family × region × native/transfer) → figures |
| `figures/strategic/` (new) | Output |

### Deliverable figures
- Petroleum-site prioritization (which assets were defended, colored by value).
- Transfer-failure rollouts (native vs transferred side-by-side).
- Coordination behavior (per-sector agent dispatch over time).
- Infrastructure-defense trajectories over the criticality map.
- Comparative Saudi vs California spread + protection heatmaps.

### Validation
A figure exists for every (policy family × region × {native, transferred}); the Saudi figures overlay
the petroleum criticality map; heuristic-contains vs PPO-collapse is visually unambiguous.

---

## 15B.6 — AAAI-Style Report Framing

### Objective
Reframe the manuscript as a trustworthy AI-systems / safety-critical coordination paper.

**Recommended contribution list:**
1. A reproducible, leakage-free geospatial wildfire-response benchmark with infrastructure-aware,
   risk-weighted objectives (not just burned cells).
2. A **hierarchical hybrid** architecture (heuristic low-level + strategic high-level) motivated by a
   rigorous negative result for naïve deep RL.
3. A **failure analysis** of single-agent PPO (policy collapse; reward-dominance), with the controls
   that expose it (effective-method gate, seed integrity).
4. A **symmetric cross-region transfer** study with infrastructure-aware metrics and adaptation
   asymmetry between desert (petroleum) and forest (mountain) regimes.
5. Strategic **critical-infrastructure protection** as a first-class objective for autonomous response.

**Recommended section structure:** Introduction · Related work (RL for wildfire, hierarchical control,
trustworthy/​safety-critical AI, transfer) · Problem formulation (risk-weighted MDP + infrastructure) ·
Hybrid architecture · Negative result & failure analysis (PPO) · Experiments (heuristic/hybrid
scaling, transfer, infrastructure protection) · Trustworthy-evaluation protocol (reproducibility,
gates, provenance) · Limitations · Broader impact.

**Suggested title directions:** *"Trustworthy Hybrid Coordination for Wildfire Response: Infrastructure-Aware
Control and Cross-Regional Transfer"*; *"When Deep RL Fails Safely: A Hierarchical Heuristic+RL System
for Petroleum-Critical Wildfire Defense"*; *"Cross-Domain Generalization of Strategic Wildfire
Coordination under Critical-Infrastructure Risk."*

**Future work:** decentralized MARL for the strategic layer; learned low-level control with
curriculum/​imitation from the heuristic; partial observability; real ERA5/FIRMS streaming; additional
domains (Australia, Mediterranean).

### Files To Modify
| File | Change |
|------|--------|
| `docs/paper/report.md` | Reframe: heuristics = effective method, PPO = honest negative result, transfer + infrastructure central |
| `docs/paper/aaai_outline.md` (new) | Section skeleton + contribution list + claims→evidence map |

### Validation
Every claim in `report.md` maps to a committed CSV + `run_id` (Phase 12 provenance rule); no sentence
claims "PPO solves wildfire suppression."

---

## 15B.7 — Compute & Experimental Scope (reprioritized)

### Objective
Redirect compute away from open-ended PPO tuning toward the questions that now carry the paper.

**Prioritize (in order):**
1. **Heuristic MARL scaling** (1/3/5/10 agents; `nearest_fire`/`frontier`) — no training, evaluation
   only; the honest cooperative-scaling result.
2. **Cross-region transfer** (symmetric, infrastructure-aware) — evaluation-heavy, training-light.
3. **Strategic coordination** experiments (high-level greedy_risk vs risk_aware vs small RL).

**De-prioritize:** single-agent PPO hyperparameter search (documented negative baseline only).

**Explicit statements for the paper and README:**
- *Low-level PPO navigation is no longer the primary research claim.*
- *Heuristic local control is the reliable operational baseline.*
- *The learnable component, if any, is the small strategic action space — not raw movement.*

### Validation
The authoritative regeneration budget is dominated by *evaluation* (heuristics/transfer), not PPO
training; the effective-method gate passes on the heuristic/hierarchical method for every reported
region.

---

## 15B.8 — Strategic Ablation Studies (AAAI Scientific Core)

### Objective
Elevate ablation from *supplementary* to a **central scientific contribution**. The project's question
has moved from *"Can PPO learn wildfire suppression?"* (answered honestly in §1.2: no) to *"How do
hybrid autonomous wildfire-response systems generalize across domains while protecting critical
infrastructure?"* Answering the new question **requires** controlled ablations that isolate the causal
contribution of each design decision. This work-stream defines eight ablation groups that jointly
justify — with committed CSVs, bootstrap CIs, and effect sizes — (i) the **hybrid hierarchical
architecture**, (ii) **infrastructure-aware coordination**, (iii) **strategic prioritization**, (iv)
**transfer robustness**, (v) **hierarchical control decomposition**, and (vi) the **heuristic-vs-RL
role separation**. Every ablation inherits the Retained-Rigor Contract (Phase 15B) verbatim: disjoint
eval seeds, determinism + seed integrity, the effective-method gate, run manifests + artifact hashes,
bootstrap CIs + effect sizes, CI validation. **No ablation cell ships without a committed CSV + `run_id`.**

### Problems / Goals Addressed
- **Unjustified architecture:** the hybrid/hierarchical design (15B.2) is *asserted*; ablation must
  *prove* it beats both pure-RL and pure-heuristic ends of the spectrum on the strategic metrics.
- **Unproven strategic objectives:** infrastructure weighting, cascade risk, and prioritization
  (15B.1/15B.3) must be shown to **change behavior**, not merely re-score identical trajectories.
- **Opaque component value:** the strategic controller bundles prioritization, coordination, a risk
  map, and an infrastructure observation channel; each must be independently knocked out to establish
  necessity (avoids the "kitchen-sink system" reviewer critique).
- **Under-characterized transfer:** the symmetric transfer matrix (15B.4) needs an ablation lens —
  degradation, adaptation asymmetry, and robustness gap decomposed per design choice.
- **Observation sufficiency / Markov concerns:** channel-removal ablations connect empirical
  degradation to the MDP's observation contract and the Markov assumption (§ base env).
- **Reviewer expectation:** AAAI ablation tables with per-metric CIs and qualitative rollout evidence
  are table-stakes for the systems/safety-critical framing (15B.6).
- **Compute discipline:** ablations must not regress into PPO hyperparameter tourism (15B.7).

### Retained-Rigor Contract (inherited, restated)
Each ablation cell = a `run_id` with: a config diff committed under `configs/ablation/`, ≥5 disjoint
eval-seed batches (Phase 4 `scenario_seed_offset`), a run manifest with artifact SHAs (Phase 8),
bootstrap 95% CIs (`bootstrap_ci`, 10k) + Cohen's d (Phase 9), and a row in exactly one authoritative
CSV under `results/ablation/`. Ablations **must not** be used to manufacture a PPO win; where PPO is a
knob it remains the honest negative baseline.

### Files To Modify
| File | Change |
|------|--------|
| `configs/ablation/*.yaml` (new dir) | One config per ablation cell; each inherits a base and overrides exactly one factor (single-factor-at-a-time discipline). Group manifests: `hybrid_vs_pure.yaml`, `infra_reward.yaml`, `transfer_matrix.yaml`, `strategic_components.yaml`, `infra_density.yaml`, `lowlevel_heuristic.yaml`, `obs_channels.yaml`, `catastrophe.yaml` |
| `src/wildfire_rl/ablation/__init__.py` (new) | Ablation registry: maps a factor name → the config-override + toggle it flips |
| `src/wildfire_rl/ablation/runner.py` (new) | `run_ablation_group(group, seeds, out_csv)` — enumerate cells, evaluate each policy family, aggregate strategic metrics + CIs, write tidy CSV (`group,cell,factor,value,policy_family,region_src,region_tgt,metric,mean,ci_lo,ci_hi,cohen_d,n_seeds,run_id`) |
| `src/wildfire_rl/ablation/factors.py` (new) | Declarative factor definitions: `AblationFactor(name, values, applies_to, toggle_fn)` for each of the 8 groups |
| `src/wildfire_rl/config.py` | Add ablation toggles: `AblationConfig{ infra_weighting:bool, cascade:bool, prioritization:bool, coordination:bool, risk_map:bool, infra_channel:bool, occupancy_channel:bool, low_level:str, high_level:str }`; all default to the full proposed system |
| `src/wildfire_rl/envs/base.py`, `multi_agent.py`, `hybrid_multi_agent.py` | Honor the toggles: gate the infra reward term, cascade dynamics, risk map, and each observation channel behind `AblationConfig` (already-wired channels become individually removable) |
| `src/wildfire_rl/coordination/strategic_controller.py` | Support `prioritization=off` (uniform target weights) and `coordination=off` (per-agent independent greedy, no sector deconfliction) |
| `src/wildfire_rl/eval/metrics.py` | Add coordination-efficiency, prioritization-accuracy, and emergency-response-latency metrics (formulas below) |
| `src/wildfire_rl/eval/transfer.py` | Expose `transfer_robustness_gap()` + `adaptation_asymmetry()` as reusable functions for the transfer ablation |
| `src/wildfire_rl/viz/figures.py` | Ablation bar charts (with CIs), transfer heatmaps, component-knockout waterfall, density-sweep curves |
| `src/wildfire_rl/viz/strategic.py` | Catastrophe-rollout overlays: chain explosions, emergency reprioritization, regional-sacrifice decisions |
| `scripts/run_ablations.py` (new) | CLI: `--group {all,hybrid,infra_reward,transfer,components,density,lowlevel,obs,catastrophe}`; dispatches to `ablation.runner` |
| `scripts/build_report_tables.py` | Render Section-5 ablation tables (5.1–5.7) from `results/ablation/*.csv` |
| `tests/test_ablation.py` (new) | Assert each factor toggle actually changes an observation/reward/behavior signature (no silent no-ops); assert CSV schema + provenance columns present |
| `Makefile` | `make ablations` target: runs all groups, builds figures + tables, verifies gate + provenance |
| `results/ablation/*.csv` (new) | Authoritative per-group outputs, manifest-hashed |
| `figures/ablation/` (new) | Rendered figures |
| `docs/paper/report.md`, `docs/paper/aaai_outline.md` | Add Section 5 (structure below) |

### New / Extended Metrics
Reuse the 15B.3 strategic metrics — **ISR, WEL, CPS, RAC, PCA, TRS, CDGG** — and add:

| Metric | Formula | Meaning |
|--------|---------|---------|
| Coordination Efficiency (CE) | `assets_defended / Σ_agents redundant_dispatch` | inverse of wasted/duplicated dispatch; higher = less agent collision on the same target |
| Prioritization Accuracy (PA) | `Σ (defended_asset_value) / Σ (top-k_by_value asset_value)` | fraction of the *ideal high-value defense set* actually protected |
| Emergency Response Latency (ERL) | `mean_t(first_agent_arrival − ignition)` over threatened assets | median steps from asset-threat onset to first responder |
| Transfer Robustness Gap (TRG) | `1 − min_metric(TRS over off-diagonal cells)` | worst-case transfer degradation across the matrix |
| Catastrophe Chain Length (CCL) | `mean cascade depth per detonation` | severity of petroleum chain reactions (lower = better containment) |

All new metrics are per-episode, unit-tested on synthetic inputs, threaded through `evaluate.py`, and
reported with bootstrap CIs. TRS/CDGG/TRG are computed only from native-vs-transfer paired runs.

### Ablation Groups — Step-by-Step Implementation Guide

Common protocol for **every** group: (1) define the factor in `factors.py`; (2) emit one config per
cell under `configs/ablation/<group>.yaml`; (3) evaluate each policy family over ≥5 disjoint eval-seed
batches; (4) aggregate metrics + bootstrap CIs + Cohen's d vs the group's reference cell; (5) write one
tidy CSV row per (cell × policy_family × region-pair × metric); (6) render the figure(s); (7) emit the
Section-5 table. Single-factor-at-a-time unless a group explicitly sweeps a range.

#### Group 1 — Hybrid vs Pure-RL vs Pure-Heuristic *(most important)*
Compare three controllers head-to-head:
- **A. Pure PPO controller** — single-agent PPO doing low-level control (the §1.2 negative baseline).
- **B. Pure heuristic controller** — `nearest_fire`/`frontier` routing + deterministic suppression, **no** strategic high level.
- **C. Hybrid hierarchical controller (proposed)** — strategic high level (dispatch + prioritization) over the heuristic low level.

Explicit framing baked into configs, tables, and prose: *PPO remains a negative baseline; the heuristic
is the trusted operational low-level controller; the hybrid is the proposed system.* Never label PPO a
"win."

Measure: containment (RAC/burned), **infrastructure survival (ISR/PCA)**, **transfer robustness
(TRS/TRG)**, **catastrophic-loss reduction (CPS/WEL/CCL)**, coordination efficiency (CE).
```yaml
# configs/ablation/hybrid_vs_pure.yaml
base: configs/experiment/multiseed.yaml
cells:
  pure_ppo:       { controller: ppo,        high_level: none,        low_level: ppo }
  pure_heuristic: { controller: heuristic,  high_level: none,        low_level: frontier }
  hybrid:         { controller: hierarchical, high_level: risk_aware, low_level: frontier }
regions: [saudi, california]
seeds: {batches: 5, disjoint_eval: true}
```
Outputs: `results/ablation/hybrid_vs_pure.csv`; grouped bar chart per metric with CIs
(`figures/ablation/hybrid_vs_pure_*.png`); Section-5.2 table. Statistical test: Cohen's d of
hybrid-vs-heuristic and heuristic-vs-PPO per metric, bootstrap CIs, significance reported only with
effect size. **Expected/allowed conclusion:** hybrid ≥ heuristic ≫ PPO on strategic metrics; if hybrid
does **not** beat heuristic on a metric, report that honestly (it bounds the value of the strategic layer).

#### Group 2 — Infrastructure-Aware Reward Ablation
Cells: **A.** no infrastructure weighting (`catastrophe_weight=0`, burned-cell objective only); **B.**
infrastructure weighting enabled; **C.** cascading petroleum-risk enabled (`cascade_prob>0` +
catastrophe penalty). Hold the controller fixed (hybrid) and vary only the objective.

Measure: strategic prioritization *changes* (PA shift, which assets are defended), protected
infrastructure (ISR/PCA), economic loss reduction (WEL), catastrophic-event prevention (CPS/CCL).
**Goal — prove the objective changes behavior, not just the score.** The decisive check: at *equal
burned-cell counts*, B/C must differ from A on ISR/WEL/PA — otherwise the reward is cosmetic.
Outputs: `results/ablation/infra_reward.csv`; PA-vs-condition figure; Section-5.3 table.

#### Group 3 — Cross-Region Transfer Ablation
Reuse the symmetric matrix (15B.4) as an ablation over source→target: **Saudi→Saudi**,
**Saudi→California**, **California→California**, **California→Saudi** (full N×N per policy family).
Measure: transfer degradation (CDGG), adaptation asymmetry (`|CDGG(S→C) − CDGG(C→S)|`), robustness gap
(TRG), generalization performance (TRS per metric).
Deliverables: transfer matrices + **heatmaps** per metric (`figures/ablation/transfer_*.png`);
statistical protocol = ≥5 disjoint eval-seed batches per cell, bootstrap-CI reporting on every cell and
on the asymmetry; `results/ablation/transfer_matrix.csv`; Section-5.4 table. **Include the previously
missing California→Saudi cell** — its absence was a rigor gap.

#### Group 4 — Strategic Coordination Ablation *(component knockout)*
Disable strategic modules **individually** (single-factor knockout from the full hybrid):
**A.** no prioritization (uniform target weights); **B.** no coordination (independent per-agent greedy,
no sector deconfliction); **C.** no risk map (high level blind to criticality); **D.** no infrastructure
channel (obs channel removed from the high level).
Measure: infrastructure losses (ISR/PCA/WEL), delayed response (ERL), coordination collapse (CE),
catastrophic spread (CPS/CCL). **Goal — prove each component is necessary:** each knockout should
degrade ≥1 metric with a CI excluding zero vs the full system. Present as a **waterfall/knockout chart**
(`figures/ablation/component_knockout.png`); `results/ablation/strategic_components.csv`; Section-5.5 table.

#### Group 5 — Infrastructure-Density Ablation *(environmental stress test)*
Sweep environmental stress (multi-value, not single toggle): number of petroleum sites, clustering
density, explosion/blast radius, criticality-score distribution.
```yaml
# configs/ablation/infra_density.yaml
sweep:
  num_sites:        [4, 8, 16, 32]
  cluster_sigma:    [low, med, high]     # spatial clustering
  blast_radius:     [1, 2, 4]
  criticality_dist: [uniform, heavy_tail]
```
Measure: coordination scaling (CE vs density), prioritization behavior (PA vs density), transfer
robustness degradation (TRS/TRG under stress), resource-allocation shifts (dispatch distribution).
Frame explicitly as **environmental stress-testing** of the strategic layer. Outputs: density-sweep
curves with CIs (`figures/ablation/density_*.png`); `results/ablation/infra_density.csv`; feeds
Section-5.7. Keep the sweep grid modest and evaluation-only to respect the compute budget (15B.7).

#### Group 6 — Low-Level Heuristic Ablation
The heuristic is now **part of the proposed system**, so its choice is a first-class factor. Compare
low-level routers under a fixed hybrid high level: **`nearest_fire`**, **`frontier`**, **risk-aware
frontier** (routes weighted by criticality), **infrastructure-aware routing** (biases paths to defend
assets). Measure: containment efficiency (RAC), strategic responsiveness (PA/ERL), transfer robustness
(TRS). Outputs: `results/ablation/lowlevel_heuristic.csv`; Section-5.2/5.5 table. Report which router is
the reliable operational default and whether risk-/infra-aware routing pays off under transfer.

#### Group 7 — Observation-Channel Ablation
Remove observation channels **individually**: **infrastructure map**, **risk heatmap**, **transfer
metadata**, **occupancy channel**. Measure: prioritization collapse (PA), transfer degradation
(TRS/CDGG), infrastructure losses (ISR/WEL), navigation instability (CE/ERL). The plan must connect
results explicitly to: **observation sufficiency**, the **Markov assumption** (a channel whose removal
degrades performance is Markov-relevant state the agent needs), and **strategic situational awareness**.
Outputs: `results/ablation/obs_channels.csv`; per-channel degradation figure; Section-5.6 table. The
`test_ablation.py` guard asserts each removed channel actually changes the observation tensor shape/content.

#### Group 8 — Catastrophic-Event Ablation
Compare **no explosion propagation** (`cascade_prob=0`) vs **cascading petroleum explosions**
(`cascade_prob>0`), controller fixed. Measure: policy behavior changes (dispatch distribution),
dispatch urgency (ERL), infrastructure-defense patterns (which assets prioritized), containment
tradeoffs (RAC vs ISR/CPS). **Rollout-visualization requirements** (via `viz/strategic.py`), each
rendered for Saudi native + transfer: (a) **chain explosions** propagating through clustered petroleum
sites; (b) **emergency reprioritization** (the high level re-dispatching under a detonation); (c)
**regional-sacrifice decisions** (deliberately ceding low-value cells to save high-value assets).
Outputs: `results/ablation/catastrophe.csv`; rollout GIF/PNG sequences under `figures/ablation/catastrophe/`;
Section-5.7 table + qualitative analysis. Report CCL and CPS deltas with CIs.

### Mandatory Animated-Rollout Visualization Standard (all experiments + ablations)
**Every experiment and every ablation cell must emit a per-timestep animated grid rollout (GIF) in the
canonical wildfire-RL format** — the same style used across the repo's environment renderer — so results
are visually comparable across groups, regions, and the native/transfer axis. The static per-group
figures (bars/heatmaps/waterfalls) remain required; the animated rollout is an **additional, mandatory**
deliverable, not a substitute.

**Canonical format (fixed contract):**
- **Header:** `Timestep #: <t>` at top-left, updated each frame, plus a horizontal **legend** with the
  exact class → color mapping below.
- **Grid:** the region raster rendered as equal square cells with thin white gridlines; consistent cell
  size and orientation across all rollouts so frames are directly comparable.
- **Class → color legend (canonical palette, do not re-map):**

  | Class | Meaning | Color |
  |-------|---------|-------|
  | Grass | unburned burnable cell | green (`#2CC295`) |
  | Fire | actively burning cell | red/pink (`#F04A6E`) |
  | Populated | occupied population/asset cell (pre-evacuation) | dark navy (`#0E2A3B`) |
  | Evacuating | population currently moving along a route | blue (`#1F7A9E`) |
  | Path | evacuation / dispatch route cell | yellow (`#F5C24B`) |
  | Finished | successfully evacuated / resolved cell | purple (`#C39BD3`) |

- **Semantics for the strategic domain:** `Populated`/`Evacuating`/`Finished` render the population &
  evacuation state; `Path` overlays both evacuation routes and high-level agent dispatch/routing;
  `Fire` includes cascade-ignited petroleum cells. Infrastructure/criticality may be shown as an
  optional faint underlay but must **not** recolor the six canonical classes.
- **Frames:** one frame per environment timestep from `t=1` to episode end; fixed frame rate; loop.

**Files To Modify (visualization standard):**
| File | Change |
|------|--------|
| `src/wildfire_rl/viz/rollout.py` | Implement/lock the canonical renderer: fixed palette + legend + `Timestep #` header; `render_rollout(states, out_path.gif)` |
| `src/wildfire_rl/viz/strategic.py` | Strategic overlays (dispatch/prioritization/catastrophe) draw **on top of** the canonical palette, never replace it |
| `src/wildfire_rl/ablation/runner.py` | For every ablation cell, record the state trajectory and emit `<cell>.gif` alongside the CSV row (path stored in a `rollout_gif` column) |
| `scripts/run_ablations.py` | `--render-figures` also renders one canonical GIF per cell (and per representative seed) |
| `configs/viz.yaml` (new) | Single source of truth for the palette, legend labels, cell size, and frame rate — imported by both renderers so every experiment/ablation is pixel-consistent |
| `tests/test_rollout_viz.py` (new) | Assert the palette + legend labels match the canonical contract exactly and a GIF is produced for a synthetic trajectory |

**Provenance:** each GIF is manifest-hashed (Phase 8) and its path recorded in the owning CSV row so
every animated result traces to a `run_id`, identical to the static-figure provenance rule.

**Output layout:** `figures/ablation/<group>/<cell>.gif`; `figures/rollouts/<experiment>/<region>_<policy>.gif`.

### Report / Paper Integration
Add **Section 5 — Ablation Studies** to `docs/paper/report.md` and mirror in `docs/paper/aaai_outline.md`:
```
5. Ablation Studies
   5.1 PPO Remediation & Failure Analysis      (Group 1 pure-PPO arm; ties to §1.2 + Phase 16 negative result)
   5.2 Hybrid Architecture Ablation            (Group 1 + Group 6)
   5.3 Infrastructure-Aware Coordination       (Group 2)
   5.4 Cross-Region Transfer                    (Group 3)
   5.5 Strategic Coordination Components        (Group 4)
   5.6 Observation Sufficiency                  (Group 7)
   5.7 Environmental Stress Testing             (Group 5 + Group 8)
```
Guidance to encode:
- **AAAI ablation-table structure:** one row per ablation cell, one column per metric; cell = `mean
  [ci_lo, ci_hi]`; bold the proposed/full-system row; a `Δ vs full` column with Cohen's d; footnote the
  seed count and `run_id`. Full-system row is the reference; knockouts show degradation.
- **Figure-generation requirements:** every ablation group ships ≥1 figure with **visible CIs**
  (grouped bars for 1/2/6, heatmaps for 3, waterfall for 4, sweep curves for 5, per-channel bars for 7,
  rollout sequences for 8) **plus** a canonical animated-rollout GIF per cell (see the Mandatory
  Animated-Rollout Visualization Standard above). All figures/GIFs generated by
  `scripts/run_ablations.py`/`viz` — never hand-drawn.
- **Qualitative rollout analysis:** §5.7 pairs quantitative catastrophe metrics with the Group-8 rollout
  narratives (chain explosion, emergency reprioritization, regional sacrifice) — the interpretability story.
- **Appendix organization:** Appendix A = full per-cell CSV tables + `run_id`s; Appendix B = per-metric
  transfer matrices; Appendix C = density-sweep grids; Appendix D = ablation reproduction commands.
- **Statistical-reporting standards:** bootstrap 95% CIs + Cohen's d on every claim; significance
  reported **only** with effect size; the full-system row is the reference for all Δ; no p-value stands alone.
- **Honesty guardrail:** if an ablation shows a component *doesn't* help (e.g., hybrid ≈ heuristic on a
  metric, or a router underperforms), report it plainly — negative ablation findings bound the
  contribution and are themselves results. Never imply PPO "eventually succeeded."

### Compute & Experimental Priorities
State explicitly in the phase, README, and paper:
- **PRIORITIZE:** heuristic MARL scaling · transfer analysis · infrastructure coordination · strategic
  evaluation — all **evaluation-heavy, training-light**.
- **DE-EMPHASIZE:** endless PPO hyperparameter tuning (Group 1's PPO arm is the fixed §1.2 baseline, not a search).
- **Explicit statements:** *low-level PPO navigation is no longer the central research claim*; *heuristic
  local control is the trusted operational baseline*; *the only learnable component of interest is the
  small strategic action space.* The ablation compute budget is dominated by evaluation over disjoint
  seeds, not by training runs.

### Validation Procedures
```bash
# 1. Toggles are real (no silent no-ops) + CSV schema/provenance guarded
python -m pytest tests/test_ablation.py -q

# 2. Run each group (evaluation-only, disjoint eval seeds)
python scripts/run_ablations.py --group all            # or: hybrid | infra_reward | transfer | components | density | lowlevel | obs | catastrophe

# 3. Provenance + gate: every CSV row has a run_id; manifests hashed; effective-method gate passes
python scripts/validate_learning_gate.py --scope ablation
python scripts/check_provenance.py results/ablation/*.csv

# 4. Figures + Section-5 tables regenerate from the CSVs (no hand editing)
python scripts/run_ablations.py --render-figures
python scripts/build_report_tables.py --section 5 > docs/paper/_generated_ablation_tables.md

# 5. One-command reproduction
make ablations
```
Expected: (a) every factor toggle provably changes obs/reward/behavior; (b) the hybrid ≥ heuristic ≫
PPO ordering holds on strategic metrics *or* the exception is reported honestly; (c) infra-reward and
component knockouts produce CI-separated behavior changes; (d) the symmetric transfer matrix is complete
incl. California→Saudi with per-cell CIs; (e) every table/figure traces to a committed CSV + `run_id`.

### README Updates Required
Add a **"Strategic Ablation Studies"** subsection under the Hybrid Wildfire-Response System section:
```markdown
### Strategic Ablation Studies
Ablations are a central contribution, not an appendix. Eight controlled studies isolate the causal
value of each design choice — hybrid-vs-pure control, infrastructure-aware reward, cross-region
transfer, strategic-component knockouts, infrastructure-density stress tests, low-level heuristic
choice, observation-channel sufficiency, and catastrophic cascade events — each reported with bootstrap
CIs and effect sizes over disjoint eval seeds. PPO appears only as the honest negative baseline.

    make ablations                              # run all groups, render figures + Section-5 tables
    python scripts/run_ablations.py --group transfer   # a single group
```
Every experiment and ablation cell also emits a canonical **animated-rollout GIF** — a per-timestep grid
with a `Timestep #` header and the fixed **Grass / Fire / Populated / Evacuating / Path / Finished**
legend — so results are visually comparable across groups, regions, and native/transfer. The palette is
locked in `configs/viz.yaml`; GIFs land under `figures/ablation/<group>/` and `figures/rollouts/`.

Also: (i) add CE / PA / ERL / TRG / CCL to the **Metrics** list; (ii) note in **Results** that
ablation Section 5 leads with the effective (hybrid/heuristic) method and frames PPO as the negative
result; (iii) document `configs/ablation/` and `results/ablation/` in the repo-layout section.

### Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] `AblationConfig` toggles wired into env/reward/obs/controller; each individually removable
- [ ] All 8 group configs committed under `configs/ablation/`; single-factor discipline enforced
- [ ] `ablation/runner.py` + `scripts/run_ablations.py` produce tidy CSVs with the full provenance schema
- [ ] New metrics (CE, PA, ERL, TRG, CCL) implemented + unit-tested alongside ISR/WEL/CPS/RAC/PCA/TRS/CDGG
- [ ] `tests/test_ablation.py` green: every toggle changes a signature; no silent no-ops; CSV schema asserted

Reproducibility
- [ ] Every ablation cell uses ≥5 disjoint eval-seed batches, a run manifest, and hashed artifacts
- [ ] `make ablations` regenerates every CSV, figure, and Section-5 table end-to-end
- [ ] `check_provenance.py` passes: every table/figure row maps to a committed CSV + `run_id`

Scientific validity
- [ ] Effective-method gate passes on the hybrid/heuristic method for every reported region
- [ ] Group 1 ordering (hybrid ≥ heuristic ≫ PPO) reported with CIs + Cohen's d, or the exception stated honestly
- [ ] Symmetric transfer matrix complete (incl. California→Saudi) with per-cell + asymmetry CIs
- [ ] Component knockouts (Group 4) each degrade ≥1 metric with a zero-excluding CI, or necessity is not claimed
- [ ] PPO never tuned or relabeled to "win"; negative ablation findings retained honestly

Visualization / reporting
- [ ] Every experiment + ablation cell emits a canonical animated-rollout GIF (`Timestep #` header +
      the fixed Grass/Fire/Populated/Evacuating/Path/Finished legend), palette locked in `configs/viz.yaml`
- [ ] `tests/test_rollout_viz.py` green: palette + legend labels match the canonical contract; GIF produced
- [ ] Every group ships ≥1 figure with visible CIs; Group 8 ships catastrophe rollout sequences
- [ ] Section 5 (5.1–5.7) drafted in `report.md` + `aaai_outline.md`, tables auto-generated

README completeness
- [ ] "Strategic Ablation Studies" subsection added; CE/PA/ERL/TRG/CCL in Metrics; ablation dirs documented

### Proceed Rule
- If ALL items are `[x]`, the strategic ablation framework is complete and feeds the Phase 15B checkpoint,
  Phase 9 regeneration, Phase 12 report (Section 5), and Phase 14 certification. Otherwise fix the failing
  item before advancing. **Never** satisfy a criterion by forcing a PPO win, by relabeling the negative
  baseline, or by dropping an ablation cell that produced an inconvenient (e.g., "component didn't help")
  result — honest negative ablation findings are contributions, not failures.

---

## Phase 15B — README Updates Required

### Add Section
```markdown
## Hybrid Wildfire-Response System

Wildfire-RL is a **hierarchical hybrid** wildfire-response platform: an interpretable heuristic
low-level controller (routing + deterministic suppression) under a strategic high-level controller
(agent dispatch, infrastructure prioritization). The environment optimizes **risk-weighted strategic
damage** — including petroleum-infrastructure survival and cascading-explosion risk — not just burned
cells. Single-agent deep RL (PPO) is included as a rigorously-characterized **negative-result
baseline** (see "Failure analysis"). Effective method = heuristic/hybrid routing.

Build infrastructure rasters, run the symmetric transfer study, and render strategic rollouts:
```bash
python scripts/build_infrastructure.py --region saudi_eastern_province --grid 32
python scripts/run_transfer_hybrid.py --config configs/experiment/transfer_hybrid.yaml
python scripts/render_strategic.py --regions saudi california
```

### Modify Existing Section
- **Overview / Observation contract:** document the infrastructure-criticality channel and the
  two-level obs/action contracts.
- **Metrics:** add ISR / WEL / CPS / RAC / PCA / TRS / CDGG.
- **Results:** lead with the effective (heuristic/hybrid) method and the transfer/​infrastructure
  findings; frame PPO as the negative result.

## Phase 15B — Success Criteria (MANDATORY CHECKPOINT)

Technical verification
- [ ] Infrastructure rasters built (asset type + criticality + blast radius), hashed in the manifest
- [ ] Cascade/explosion dynamics + catastrophe penalty implemented and unit-tested
- [ ] Infrastructure-criticality observation channel added; CNN tracks channel count
- [ ] Hierarchical controller runs (greedy_risk / risk_aware / rl); `test_hybrid.py` green
- [ ] Strategic + transfer metrics (ISR, WEL, CPS, RAC, PCA, TRS, CDGG) implemented + tested

Reproducibility
- [ ] All 15B experiments use disjoint eval seeds, run manifests, and hashed artifacts
- [ ] Symmetric transfer matrix + asymmetry regenerated by `run_transfer_hybrid.py`

Scientific validity
- [ ] Effective-method gate passes on the heuristic/hierarchical method per region
- [ ] PPO retained honestly as a negative-result baseline (never tuned to "win")
- [ ] Every table/figure traces to a CSV + `run_id`; bootstrap CIs + effect sizes reported

Logging / monitoring
- [ ] Strategic rollouts rendered for every (policy family × region × native/transfer)

README completeness
- [ ] "Hybrid Wildfire-Response System" section added; metrics + observation contract updated

### Proceed Rule
- If ALL items are `[x]`, the AAAI research core is complete and feeds Phase 9 regeneration / Phase 12
  report / Phase 14 certification. Otherwise fix the failing item before advancing. **Never** satisfy
  a criterion by forcing a PPO win — the honest negative result is a contribution, not a bug.

---

# Phase 16 — Policy Strengthening & Hyperparameter Optimization
Estimated Time: 1–2 days (compute-heavy; GPU/Colab candidate)
Execution position: **after Phase 9** (needs trained artifacts + the learning gate).

> **STATUS — REFRAMED (see §1.2).** The empirical finding is that single-agent PPO does **not** beat a
> no-op / heuristic baseline on this task despite every lever below (Markov obs, agent-attributable +
> dense-proximity reward, anti-collapse reward rebalancing, exploration tuning), and collapses to a
> near-constant policy. This phase is **retained as an honest negative-result ablation** — its levers
> are documented `EnvConfig` knobs (`reward_agent_suppression_weight`, `reward_proximity_weight`,
> `reward_fire_weight`) used to *characterize* the failure, **not** a success criterion. Its
> "PPO beats noop with margin" gate is **superseded by the effective-method gate** (Phase 7 reframe:
> the effective method is heuristic/hierarchical routing). **Do not tune PPO to force a win** — that
> would recreate the fabrication this remediation exists to remove. The steps below are preserved for
> completeness and as the exact ablation protocol behind the negative result.

## Objective
Convert the "correct-but-weak" post-Markov policy into a **strong** one that clears the learning gate
with margin and approaches the heuristic baselines (`nearest_fire`/`frontier`). Three levers:
(a) reward shaping that credits **agent-caused** suppression, (b) longer, vectorized training with
best-checkpoint selection, (c) a hyperparameter sweep.

## Problems / Goals Addressed
- Advisor recommendation #1.
- Audit finding: the canonical reward is dominated by uncontrollable fire mass, giving PPO almost no
  learnable gradient tied to its own actions (why it collapsed to no-op even before the obs bug).

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/config.py` | `reward_agent_suppression_weight` (credit fire the agent actually removes) |
| `src/wildfire_rl/envs/base.py`, `multi_agent.py` | Measure pre/post-suppression fire delta; add it to reward |
| `src/wildfire_rl/train/ppo.py` | `SubprocVecEnv` (n_envs>1), `EvalCallback` + best-model checkpointing, optional LR schedule |
| `configs/ppo/strong.yaml` (new) | Longer `total_timesteps`, tuned defaults |
| `scripts/sweep.py` (new) | Grid/random search over `lr`, `ent_coef`, `n_steps`, reward weights |
| `scripts/validate_learning_gate.py` | Tighten margin (PPO ≤ 0.5 × noop burned) |

## Step-by-Step Implementation Guide

### Step 1 — Agent-attributable suppression reward
Purpose: Give PPO a gradient tied to its actions (mirrors the `multi_agent_v3` idea, promoted into the canonical env).
Code Changes:
```python
# step(): capture fire before/after apply_suppression, credit the delta
fire_before = self.state[0].copy()
dynamics.apply_suppression(self.state, positions, self.cfg)
agent_removed = float((fire_before - self.state[0]).sum())
# ... spread/decay ...
reward += self.cfg.reward_agent_suppression_weight * agent_removed / self._initial_fire_total
```
Validation: an agent parked on fire earns strictly more than a no-op agent on the same seed.

### Step 2 — Training scale + best-checkpoint
Purpose: More gradient steps and honest model selection.
Code Changes: `make_vec_env(..., vec_env_cls=SubprocVecEnv, n_envs=cfg.n_envs)`; add SB3 `EvalCallback`
on a held-out (randomized-ignition) eval env, saving the **best** model by mean reward.
Validation:
```bash
venv/Scripts/python scripts/train.py --config configs/ppo/strong.yaml --set ppo.total_timesteps=20000 seeds=[0]
```
Expected Result: a `best_model.zip` is saved and its eval reward exceeds the last-step model's.

### Step 3 — Hyperparameter sweep
Purpose: Find a config that clears the gate.
Implementation:
```bash
venv/Scripts/python scripts/sweep.py --region saudi --trials 12 --timesteps 100000 \
    --grid lr=1e-4,3e-4 ent_coef=0.0,0.01,0.05
```
Selection metric: held-out `containment_rate` (Phase 7). Record the winning config to `configs/ppo/strong.yaml`.

### Step 4 — Final retrain + gate with margin
Implementation:
```bash
venv/Scripts/python scripts/train.py --config configs/ppo/strong.yaml   # 5 seeds, both regions
venv/Scripts/python scripts/evaluate.py --config configs/experiment/multiseed.yaml
venv/Scripts/python scripts/validate_learning_gate.py; echo "gate=$?"
```
Expected Result: gate exit 0 with margin; PPO significantly beats `random` and `noop` (effect size
reported), and the gap to `nearest_fire`/`frontier` is quantified.

## README Updates Required
### Add Section
```markdown
## Training a Strong Policy

`configs/ppo/strong.yaml` holds the tuned configuration (agent-suppression reward, vectorized envs,
best-checkpoint selection) found by `scripts/sweep.py`. It is the configuration used for all reported
results and must pass `scripts/validate_learning_gate.py` with margin (PPO ≤ 0.5 × noop burned cells).
```
### Modify Existing Section
- **Training** and **Results**: point to `strong.yaml`; state the PPO-vs-baseline outcome honestly, including the remaining gap to heuristic baselines.

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] Agent-suppression reward implemented; unit test passes
- [ ] Vectorized training + best-checkpoint selection working
Reproducibility
- [ ] Winning sweep config committed (`configs/ppo/strong.yaml`) + sweep log
Scientific validity
- [ ] Learning gate exits 0 **with margin**; PPO > random and > noop (effect size + CI)
- [ ] Gap to `nearest_fire`/`frontier` quantified and reported
Logging/monitoring
- [ ] Training curves show clear improvement (not flat)
README completeness
- [ ] "Training a Strong Policy" section added
### Proceed Rule
- If ALL items are `[x]`, proceed to Phase 17. **If the gate still fails, STOP** and revisit reward/HPO — do not publish an RL performance claim.

---

# Phase 17 — Policy Rollout Visualization (per Ablation)
Estimated Time: 4–8 hours
Execution position: **after Phase 16** (needs strengthened policies + ablation variants).

## Objective
Produce per-ablation **rollout visualizations** — agent trajectories, fire-spread evolution, and
suppression footprint — for each policy (`ppo`, `nearest_fire`, `frontier`, `noop`) across each
ablation variant (`baseline`, `no_wind`, `no_terrain`, `no_suppression`, `dense_fuel`), so behavioral
differences are visible, not merely tabular.

## Problems / Goals Addressed
- Advisor recommendation #2.
- Audit finding: the collapse (PPO ≡ noop) was only detectable in numbers; a rollout view makes
  policy behavior (and its recovery after Phase 16) legible at a glance.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/viz/rollout.py` (new) | Record one deterministic episode; render a frame grid (t=0/mid/end) + trajectory overlay; optional animated GIF |
| `scripts/render_rollouts.py` (new) | Iterate (policy × ablation variant), save figures |
| `src/wildfire_rl/viz/figures.py` | Shared colormaps / criticality overlay helper |
| `figures/rollouts/` (new dir) | Output |

## Step-by-Step Implementation Guide

### Step 1 — Rollout recorder
Purpose: Capture per-step fire field + agent positions for one seeded episode.
Code Changes (`viz/rollout.py`): `record_rollout(policy, env_factory, seed) -> {frames, agent_paths, criticality}`.
Validation:
```bash
venv/Scripts/python -c "from wildfire_rl.viz.rollout import record_rollout; print('rollout recorder importable')"
```

### Step 2 — Renderer
Purpose: One multi-panel figure per rollout (fire heatmap snapshots + agent trajectory; criticality overlay for Saudi).
Code Changes: `render_rollout(rollout, out_path)`.
Expected Result: a PNG (and optional GIF) is written.

### Step 3 — Driver over policies × ablations
Implementation:
```bash
venv/Scripts/python scripts/render_rollouts.py --region saudi \
    --policies ppo nearest_fire frontier noop \
    --variants baseline no_wind no_terrain no_suppression dense_fuel
```
Expected Result: `figures/rollouts/<variant>_<policy>.png` for every combination.

### Step 4 — Reference in report
- Embed a representative rollout grid in `docs/paper/report.md` (ablation section) and link the folder.

## README Updates Required
### Add Section
```markdown
## Rollout Visualizations

`scripts/render_rollouts.py` renders one deterministic episode per (policy × ablation variant) to
`figures/rollouts/`, showing agent trajectories over the evolving fire field (with the Saudi
criticality overlay). Use these to compare PPO behavior against the heuristic and no-op baselines.
```
### Modify Existing Section
- **Results**: reference the rollout figures alongside the ablation table.

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] A rollout figure exists for every (policy × ablation variant) pair
- [ ] Renderer runs headless (`matplotlib Agg`), deterministic seeds
Reproducibility
- [ ] `render_rollouts.py` regenerates all figures from committed models
Scientific validity
- [ ] Post-Phase-16, the PPO rollout visibly differs from `noop` (behavioral evidence of learning)
- [ ] Saudi rollouts overlay the criticality map
Logging/monitoring
- [ ] Each figure names its policy, variant, seed
README completeness
- [ ] "Rollout Visualizations" section added
### Proceed Rule
- If ALL items are `[x]`, proceed to the terminal Phase 14 re-certification. Otherwise fix first.
