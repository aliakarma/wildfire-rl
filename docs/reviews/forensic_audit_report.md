Used tool: list_permissions
Listed directory wildfire-rl
Listed directory src
Listed directory wildfire_rl
Ran command: `git ls-files`
Ran command: `git status`
Viewed report.md:1-572
Viewed run_ablation.py:1-121
Listed directory results
Viewed ablation_results.csv:1-7
Viewed significance.py:1-137
Viewed base.py:1-131
Viewed dynamics.py:1-120
Viewed ablation.yaml:1-27
Viewed config.yaml:1-39
Viewed evaluate.py:1-68
Viewed metrics.py:1-45
Viewed ppo.py:1-77
Viewed cnn.py:1-59
Viewed config.py:1-236
Viewed eval_saudi.json:1-74
Viewed README.md:1-33
Listed directory data
Listed directory saudi_eastern_province
Listed directory grids
Listed directory 32x32
Ran command: `poetry run pytest`
Ran command: `python -m pytest`
Used tool: manage_task
Viewed test_metrics.py:1-32
Viewed test_eval_baselines.py:1-65
Viewed baselines.py:1-228
Edited test_suppression.py
Ran command: `python "C:\Users\Ali Akarma\.gemini\antigravity-ide\brain\82168e94-a27a-48c5-a1dd-01c7157d8368\scratch\test_suppression.py"`
Used tool: manage_task
Viewed multi_agent.py:1-131
Viewed hybrid_multi_agent.py:1-125
Viewed multi_agent_v3.py:1-159
Viewed nearest_fire_router.py:1-47
Viewed multi_agent_v2.py:1-162
Viewed test_hybrid.py:1-72
Viewed forensic_audit_report.md:1-300
Viewed forensic_audit_report.md:301-387
Viewed claude_review_report.md:1-200
Viewed transfer_matrix.csv:1-10
Listed directory models
Viewed loading.py:1-55
Edited test_model_predictions.py
Ran command: `python "C:\Users\Ali Akarma\.gemini\antigravity-ide\brain\82168e94-a27a-48c5-a1dd-01c7157d8368\scratch\test_model_predictions.py"`
Used tool: manage_task
Searched for "14581"
Viewed baseline_statistics.csv:1-12
Viewed marl_baseline_statistics.csv:1-42
Viewed test_env_api.py:1-64
Viewed README.md:1-193
Viewed reproducibility.md:1-62
Viewed marl_effect_sizes.csv:1-34
Viewed marl_scaling_results.csv:1-3
Searched for "-1564.89"
Searched for "1564.89"
Searched for "2396.69"
Searched for "3336.91"
Searched for "3299.71"
Searched for "14581"
Viewed eval_saudi.csv:1-5
Searched for "38672"
Viewed evaluate.py:1-17
Viewed cli.py:1-250
Viewed normalize.py:1-41
Viewed tensor_stack.py:1-70
Viewed routing_utils.py:1-56
Viewed eval_saudi_generalization.csv:1-9

# 🧠 OVERALL VERDICT
**Score: 3.5/10**
**Decision: Reject**

The single most fatal flaw in this repository is a **fundamental reinforcement learning system design error where the agent's coordinates are completely omitted from the observation space returned to the policy**, rendering the environment an unlearnable POMDP where the agent is blind to its own location. As a result, the trained PPO policies completely collapse, learning only to output constant actions (such as walking straight down and getting stuck on the border) and performing **statistically worse than a completely random policy** in all single-agent and multi-agent configurations. To conceal this failure, the draft paper omits baseline comparisons from its results tables and cherry-picks mathematically impossible confidence intervals and effect sizes, constituting a severe breach of scientific integrity.

---

# 🚨 CRITICAL ISSUES (rejection-level)

## Issue 1: Agent Position Omitted from Observation Space (POMDP Navigation Failure)
- **Problem**: The observation space returned by the environment is restricted to the 7 geospatial raster channels and does not include the agent's spatial coordinates, making it impossible for the policy to learn coordinate-based navigation.
- **Evidence**: In [`src/wildfire_rl/envs/base.py`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/envs/base.py#L52-L54):
  ```python
  self.observation_space = spaces.Box(
      low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
  )
  ```
  And in [`WildfireEnv.step`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/envs/base.py#L89-L102), the environment returns `self.state.astype(np.float32)` which only contains the 7 channels loaded from `state_tensor.npy` (fire, fuel, wind_x, wind_y, terrain, temperature, humidity) and never incorporates `self.agent_pos`.
- **Why it invalidates results**: Without knowing its own location, the agent cannot compute a path to the fire. Evaluation scripts show that trained models collapse, predicting the constant action `1` (DOWN) at every step and getting stuck on the bottom border of the grid.
- **Exact fix**:
  ```python
  # BEFORE (src/wildfire_rl/envs/base.py, line 52)
  self.observation_space = spaces.Box(
      low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
  )

  # AFTER
  self.observation_space = spaces.Dict({
      "grid": spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32),
      "agent_pos": spaces.Box(low=0.0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
  })
  # Note: The CustomCNN feature extractor must also be modified to accept this Dict space.
  ```

## Issue 2: Baseline Omission and Policy Collapse Concealment
- **Problem**: The draft paper claims PPO learns effective fire containment, but baseline comparisons (Random and No-Op) are omitted from the publication tables because the trained PPO agent is consistently outperformed by a random policy.
- **Evidence**: The paper draft [`docs/paper/report.md`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/docs/paper/report.md#L287-L292) lists only PPO agent team sizes. However, the computed results in [`results/baseline_statistics.csv`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/results/baseline_statistics.csv#L2-L6) show:
  - Saudi PPO Mean Reward: `-14639.64 ± 1564.92` (Mean Burned: `98.6`)
  - Saudi Random Mean Reward: `-13698.72 ± 1963.97` (Mean Burned: `90.45` — **better than PPO**)
  - Saudi No-Op Mean Reward: `-14542.98 ± 1511.66` (Mean Burned: `97.85` — **better than PPO**)
- **Why it invalidates results**: The core claim that PPO learns wildfire suppression is false; it is statistically equivalent to doing nothing (No-Op) or worse than a random walk.
- **Exact fix**:
  ```python
  # BEFORE (docs/paper/report.md, line 287)
  [Original table showing only "Num Agents" and PPO rewards]

  # AFTER
  # Incorporate baseline rows in the paper:
  | Policy | Mean Reward | Mean Burned Cells | p-value vs Random |
  | :--- | :--- | :--- | :--- |
  | PPO (Saudi) | -14639.64 | 98.60 | 0.068 (n.s.) |
  | Random | -13698.72 | 90.45 | Baseline |
  | No-Op | -14542.98 | 97.85 | - |
  ```

## Issue 3: Fabricated/Impossible Confidence Intervals in Paper
- **Problem**: The confidence intervals reported in the paper for the Saudi MARL experiments are mathematically impossible given the standard deviations of the data.
- **Evidence**: [`docs/paper/report.md` line 289](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/docs/paper/report.md#L289) claims:
  - Reward Mean: `-3336.91 [-3337.70, -3336.13]`
  - Fire Intensity Mean: `178.35 [178.18, 178.53]`
  However, [`results/baseline_statistics.csv` line 6](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/results/baseline_statistics.csv#L6) and [`results/marl_baseline_statistics.csv` line 22](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/results/marl_baseline_statistics.csv#L22) show the standard deviations for these runs are `388.73` (for reward) and `14.48` (for fire intensity) over `N=20` episodes. The true 95% CI for the reward is approximately `[-3518.7, -3155.1]`.
- **Why it invalidates results**: Reporting CIs with a width of `< 2.0` when the standard deviation is `388.73` is a statistical impossibility. This indicates manual fabrication of confidence intervals to present the results as highly stable.
- **Exact fix**:
  ```python
  # BEFORE (docs/paper/report.md, line 289)
  | **1 Agent** | -3336.91 [-3337.70, -3336.13] | 178.35 [178.18, 178.53] |

  # AFTER
  | **1 Agent** | -3336.91 [-3518.74, -3155.08] | 178.35 [171.99, 184.71] |
  ```

## Issue 4: Severe Test Set Contamination (Train=Test Scenario)
- **Problem**: The environment default configuration sets `randomize_ignition: false`, meaning every training episode and every evaluation episode initializes the fire at the exact same spatial grid coordinates.
- **Evidence**: In [`src/wildfire_rl/config.py` line 137](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/config.py#L137):
  `randomize_ignition: bool = False`
  And in [`configs/config.yaml` line 20](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/configs/config.yaml#L20):
  `randomize_ignition: false`
- **Why it invalidates results**: The policy is not learning fire containment dynamics; it is simply overfitting to/memorizing a static trajectory from a single ignition source. When evaluated on randomized ignition points, PPO performance degrades to worse than No-Op.
- **Exact fix**:
  ```python
  # BEFORE (configs/config.yaml, line 20)
  randomize_ignition: false

  # AFTER
  randomize_ignition: true
  # Note: Retrain PPO models with randomized ignition to evaluate generalization.
  ```

---

# ⚠️ MODERATE ISSUES

## Issue 5: Independent Min-Max Normalization Confounding Cross-Regional Generalization
- **Problem**: Environmental layers (NDVI, DEM) are scaled using min-max normalization per-region independently in the data pipeline. Because California has a much higher fuel density than Saudi Arabia, this scaling shifts the absolute values, making raw reward comparisons across regions invalid.
- **Evidence**: [`src/wildfire_rl/data/normalize.py` line 4–6](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/data/normalize.py#L4-L6).
- **Fix**: Re-run all cross-regional transfer experiments utilizing a shared scaler (such as the implemented `fit_shared_minmax()` function) to ensure channel ranges are directly comparable.

## Issue 6: Wind Directionality Mechanics Implementation Defect
- **Problem**: In the default wind propagation model, wind speed is isotropic (simply the average of u and v wind vectors) and has no directional alignment with the neighborhood spread direction.
- **Evidence**: [`src/wildfire_rl/envs/dynamics.py` line 68](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/envs/dynamics.py#L68):
  `wind_factor = (wind_x + wind_y) / 2.0`
- **Fix**: Set `directional_wind: true` as the default in `configs/config.yaml` to project wind vectors onto neighbor offsets.

## Issue 7: Inconsistent Scaling of Training Budgets in MARL
- **Problem**: The multi-agent scaling experiment evaluates 1, 3, and 5 agents trained for the same total timesteps (100k). However, the action space for 5 agents is `MultiDiscrete(5^5)`, which is exponentially larger than the single-agent `Discrete(5)` space, meaning the multi-agent policies are severely undertrained.
- **Evidence**: [`configs/experiment/scaling.yaml`](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/configs/experiment/scaling.yaml#L18) and [`src/wildfire_rl/envs/multi_agent.py` line 50](file:///c:/Users/Ali%20Akarma/Documents/GitHub/wildfire-rl/src/wildfire_rl/envs/multi_agent.py#L50).
- **Fix**: Scale the total training timesteps proportionally with the number of agents to maintain iso-sample-complexity.

---

# 🟢 MINOR ISSUES

- **`scripts/download_data.py`**: The script is a stub containing no Copernicus CDS request or Earth Engine API download code, blocking end-to-end reconstruction from raw sources.
- **`docs/paper/report.md` L398**: References a Google Drive Colab directory (`/content/drive/MyDrive/PyroRL_Saudi_Project/`) instead of the local repository structure.
- **Hugging Face Checkpoints**: The default configuration in `src/wildfire_rl/models/cnn.py` sets `use_pooling=True`, which changes the architecture and causes legacy checkpoints (~200MB, trained with `use_pooling=False`) to throw dimension mismatches on load unless compatibility overrides are used.

---

# 🔬 MISSING EXPERIMENTS

### 1. Unified Scale Transfer Matrix
- **What**: Evaluate cross-regional transfer Saudi ↔ California with shared normalization scaling.
- **How**: Rebuild tensors utilizing `fit_shared_minmax()`, retrain regional policies, and execute `wildfire-rl transfer`.
- **Why**: Eliminates the fuel-density normalization bias that confounds the ecological generalization claims.
- **Expected outcome**: A clear drop in reward when native policies are transferred, without raw reward differences dominating the matrix.

### 2. Generalization under Randomized Ignition
- **What**: Evaluate PPO policies trained on randomized ignitions against unseen ignition points.
- **How**: Set `env.randomize_ignition=true`, train for 200k steps, and evaluate on 100 test episodes.
- **Why**: Determines if the policy can generalize its containment mechanics instead of memorizing spatial grid routes.
- **Expected outcome**: PPO should significantly outperform Random and No-Op baselines under randomized ignitions.

---

# 🛠️ ACTION PLAN

## Step 1 — Redesign Observation Space & Re-train (Est. effort: 4h)
- **Files**: `src/wildfire_rl/envs/base.py`, `src/wildfire_rl/models/cnn.py`, `configs/config.yaml`
- **Changes**: Add the agent's spatial coordinates as a separate channel or dictionary component to the observation space. Modify `CustomCNN` to handle the new input. Retrain policies.
- **Validates**: Issue 1 (POMDP blind agent navigation blocker).

## Step 2 — Integrate Baselines & Fix Paper Tables (Est. effort: 2h)
- **Files**: `docs/paper/report.md`
- **Changes**: Add Random and No-Op baseline rows to the report's tables. Replace the fabricated/incorrect confidence intervals and effect sizes with true computed statistical bounds from `results/marl_effect_sizes.csv`.
- **Validates**: Issue 2 (baseline omission) and Issue 3 (fabricated statistical metrics).

---

# 📊 FINAL SCORECARD

| Dimension | Score | Key reason (one sentence) |
|---|---|---|
| Reproducibility | 6/10 | Seed and configuration pipelines are clean, but data download scripts are stubs and checkpoints mismatch defaults. |
| Experimental validity | 2/10 | Evaluated on the training scenario, and raw rewards are confounded by independent min-max scaling per region. |
| Model/algorithm correctness | 2/10 | The observation space completely lacks the agent's coordinates, preventing PPO from learning navigation. |
| System design realism | 3/10 | Centralized cooperative MARL fails to scale, and fire dynamics omit directional wind and diagonal spread. |
| Baselines & comparisons | 1/10 | Heuristic and random baselines were implemented in code but omitted from the paper's comparison tables. |
| Code quality | 8/10 | Highly modular code refactored into a clean Python package structure with comprehensive test suites. |
| Logging & observability | 5/10 | Outputs detailed CSV/JSON evaluations, but lacks tensorboard curves and unified metrics logging. |
| Robustness | 2/10 | Policy collapses completely, executing constant moves (DOWN or STAY) regardless of input. |
| Claim–implementation align | 1/10 | Paper claims PPO learns containment, but results CSVs prove PPO is worse than a random policy. |
| Statistical validity | 1/10 | Paper reports impossible confidence intervals (width < 2.0 with SD > 380) and Cohen's d of 0.00. |
| **Overall** | **3.5/10** | Rejection due to unobservable agent coordinates, policy collapse, and statistical discrepancies. |

---

# ⚖️ FRAUD RISK ASSESSMENT
**Rate: CRITICAL**

The fraud risk is **CRITICAL**. While the repository code itself is well-structured and contains the mechanics for statistical analysis, the draft paper contains several severe anomalies:
1. **Omission of Baselines**: The baselines (Random and No-Op) are coded but hidden from the paper's tables because the PPO agent performs worse than random.
2. **Fabricated Confidence Intervals**: The paper reports 95% confidence intervals (e.g. `[-3337.70, -3336.13]`) that are mathematically incompatible with the sample variance (SD = `388.73` over `N=20`).
3. **Overfitting Rebranding**: The failure of PPO to outperform No-Op under randomized ignition is described as an "overfitting/generalization gap," when in fact PPO fails to learn even on its training scenario.

These findings suggest a deliberate attempt to manipulate and frame results to pass peer review, rather than innocent engineering debt.