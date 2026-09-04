# Benchmark regimes (Week-6 / Phase-1 revision)

The **regime** selects the fire-scenario conditions the environment runs under. It exists
because the original benchmark used one extreme ERA5 fire day (FFMC ~96, wind ~21 km/h,
FWI ~30) with 60-simulated-minute action steps, in which the fire overruns the whole landscape
regardless of what agents do — so every policy tied the No-Op baseline. The regimes below define
**suppression-relevant** conditions where skilled coordination measurably protects
infrastructure.

**Source of truth:** `src/wildfire_marl/env/regimes.py` (`REGIMES` + per-region overrides).
These YAML files do not redefine the physics knobs — they are *training configs* that select a
region + regime and set the learning budget. Anything a config sets (e.g. `steps_per_action`)
overrides the regime default.

## Regimes

| regime | weather (vs ERA5) | step (sim-min) | ignition | firebreak | purpose |
|---|---|---|---|---|---|
| `legacy` | extreme (unchanged) | 60 | uniform | none | reproduce original (unsuppressible) results |
| `default` | moderated | 30 | threat (upwind of assets) | 3×3 | **main benchmark** (== `medium`) |
| `easy` / `medium` / `hard` | increasing severity | 20 / 30 / 40 | threat | 3×3 | difficulty axis for the robustness study |

Region-specific values (Saudi vs California differ in fuel/geography) live in
`REGION_OVERRIDES` in `regimes.py`. **Cell2Fire FBP spread physics is never modified** — only
scenario weather inputs, ignition placement, and the wrapper firebreak footprint change.

## Usage

In a training config (`configs/marl/...yaml`):

```yaml
region: saudi        # or california
regime: default      # legacy | easy | medium | hard | default
algo: mappo          # mappo | qmix | commnet | hier_comm (Phase 2+)
total_steps: 100000
```

Or from the eval harness:

```bash
python scripts/run_phase4_ablations.py --study robustness --regions saudi \n    --ckpt-dir results/wildfire_phase3_multiseed --episodes 15 --out /tmp/regime_eval
```

Programmatically:

```python
from wildfire_marl.env.regimes import make_marl_env
env = make_marl_env("saudi", regime="default")          # one env definition for all methods
env = make_marl_env("california", regime="hard", num_agents=6)  # overrides win
```
