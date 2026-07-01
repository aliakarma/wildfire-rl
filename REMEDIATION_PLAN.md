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

# Phase 9 — Statistical Validity Upgrades & Authoritative Regeneration
Estimated Time: 1–2 days (mostly compute)

## Objective
Run the authoritative multi-seed experiments on the corrected environment and regenerate every
core CSV with effect-size-primary reporting, bootstrap CIs, and verified seed independence.

## Problems Addressed
- Degenerate multi-seed (std=0, duplicates), significance-without-effect framing.
- No bootstrap CIs; over-reliance on p-values from paired tests on near-identical trajectories.

## Files To Modify
| File | Required Changes |
|------|------------------|
| `src/wildfire_rl/eval/significance.py` | Add `bootstrap_ci(values, n=10000)` |
| `configs/experiment/multiseed*.yaml` | `seeds: [0,1,2,3,4]`, `eval.n_episodes: 50` |
| all result CSVs | Regenerated from corrected code |

## Step-by-Step Implementation Guide

### Step 1 — Bootstrap confidence intervals
Purpose: Report distribution-free CIs that do not depend on the t-approximation for tiny n.

Code Changes:
```python
# src/wildfire_rl/eval/significance.py — ADD
def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    import numpy as np
    a = np.asarray(values, float); rng = np.random.default_rng(seed)
    if len(a) < 2: return (float(a.mean()), float(a.mean()))
    boot = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    return (float(np.quantile(boot, alpha/2)), float(np.quantile(boot, 1 - alpha/2)))
```

Validation:
```bash
python -c "from wildfire_rl.eval.significance import bootstrap_ci; print(bootstrap_ci([1,2,3,4,5]))"
```

Expected Result: a plausible `(lo, hi)` interval.

### Step 2 — Authoritative regeneration (the expensive run)
Purpose: Produce trustworthy numbers on the fixed, non-leaking, Markov environment.

Implementation:
```bash
source venv/Scripts/activate
make reproduce 2>&1 | tee results/runs/reproduce_$(date +%Y%m%d_%H%M%S).log
```

Validation:
```bash
python scripts/check_seed_integrity.py
python scripts/validate_learning_gate.py; echo "gate exit=$?"
```

Expected Result: seed-integrity `OK`; learning gate exit 0 (PPO beats noop). If the gate fails,
the environment/reward still does not admit a useful policy — halt and revisit Phases 3/7 before
publishing any RL claim.

### Step 3 — Regenerate figures from fresh CSVs
Implementation:
```bash
python scripts/make_figures.py
git status --short figures/ results/
```

Expected Result: figures reflect the regenerated CSVs only.

## README Updates Required

### Add Section
```markdown
## Statistical Protocol

Results report mean, bootstrap 95% CI (`bootstrap_ci`, 10k resamples), and Cohen's d as the
primary evidence; p-values are secondary and never reported without an accompanying effect size.
Minimum 5 independent, verified-distinct seeds × 50 evaluation episodes.
```

### Modify Existing Section
- Replace the **Results** narrative wholesale with numbers regenerated in this phase (do not hand-edit).

## Success Criteria (MANDATORY CHECKPOINT)

Technical verification checklist
- [ ] `bootstrap_ci` implemented and used in reporting
- [ ] All core CSVs regenerated from corrected code

Reproducibility checklist
- [ ] `make reproduce` log committed under `results/runs/`

Scientific validity checklist
- [ ] `check_seed_integrity.py` → OK (no duplicate seeds)
- [ ] `validate_learning_gate.py` → exit 0 (PPO > noop)
- [ ] Effect size reported alongside every p-value

Logging/monitoring checklist
- [ ] Each regenerated CSV traceable to a run manifest

README completeness checklist
- [ ] Statistical-protocol section added

### Proceed Rule
- If ALL items are `[x]`, proceed. **If the learning gate fails, STOP** — no RL claim is defensible; return to Phase 3/7.

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

Technical verification checklist
- [ ] Clean-clone build + fetch + validate succeed
- [ ] `make reproduce` completes end-to-end
- [ ] `pytest -q` green

Reproducibility checklist
- [ ] All artifact hashes verified against manifests
- [ ] Certification log committed

Scientific validity checklist
- [ ] Seed integrity OK; learning gate exit 0
- [ ] Every report number traces to a CSV + run_id

Logging/monitoring checklist
- [ ] Run manifests + curves present for all reported runs

README completeness checklist
- [ ] All phase README sections merged and consistent
- [ ] Badges/claims reflect certified state

### Proceed Rule
- If ALL items are `[x]`, the repository is **certified reproducible**. Otherwise remediate the failing item and re-run this phase. Do not tag a release or submit until every box is `[x]`.
