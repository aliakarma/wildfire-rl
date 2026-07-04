# REMEDIATION_PLAN_V2 — Road to AAAI Main Track

**Project:** Decentralized Multi-Agent Wildfire Suppression with Critical-Infrastructure Protection and
Cross-Regional Transfer, on a **validated fire simulator** (Cell2Fire).

**Purpose of this document.** `REMEDIATION_PLAN.md` (V1) took the project from a fabricated/flawed repo
to an *honest, reproducible* state. But V1 is built on a **hand-rolled toy simulator** with **oracle
heuristics** and a **stubbed learned controller** — three reasons an AAAI reviewer rejects on sight.
This V2 plan re-founds the research on a **peer-reviewed, physics-validated environment** (Cell2Fire,
via the Firehose RL wrapper), adds a **genuine learned contribution** (decentralized cooperating MARL),
grounds **critical infrastructure** in real data, and runs a **symmetric Saudi↔California transfer**
study. Completing all phases yields a submission that is *defensible at the AAAI main technical track*.

**The single research bet (state it honestly, everywhere).** The paper stands or falls on one claim:
*a learned, coordinated, infrastructure-aware MARL policy beats strong heuristics and naïve deep RL on
validated fire physics.* Firehose independently reports that plain PPO **fails to train** on larger,
random-ignition suppression — so beating that failure with a *coordinated hierarchical* method is a
real contribution, but it is **not guaranteed**. Phase 8 is a hard go/no-go gate; if learning does not
beat heuristics, the honest fallback is a benchmark/negative-result paper (workshop tier).

---

## What carries over vs. what is retired

**Carries over (your strongest V1 assets — reused, not rebuilt):**
- Reproducibility & provenance: run manifests, artifact hashing, `check_seed_integrity`, disjoint eval
  seeds, bootstrap CIs + Cohen's d, CI `validate` job, `build_report_tables` (claims→CSV provenance).
- Transfer methodology: symmetric matrix, TRS/CDGG, adaptation asymmetry, `scenario_seed_offset`.
- Strategic-metric design: ISR / WEL / CPS / RAC / PCA / PA / CCL / CE (re-anchored to the new env).
- The hierarchical strategic-controller *idea* and the coordination-ablation design.
- The honest negative-result framing and effective-method gate.

**Retired (the weak, un-credible parts):**
- The custom suppression simulator (`envs/dynamics.py`, `envs/base.py`, `multi_agent*.py`) — replaced by
  Cell2Fire physics.
- Oracle heuristics that read `env.agent_pos` (the privilege confound) — replaced by
  information-matched policies.
- The stubbed `rl` high-level — replaced by a genuinely trained hierarchical MARL policy.
- Invented infrastructure parameters — replaced by real asset data + sensitivity analysis.

---

## Compute, platform & tooling requirements (read before Phase 0)

- **Platform: Linux (native, WSL2, or a Linux VM / Colab / cluster).** Cell2Fire is a C++ build (Eigen +
  `make`); Firehose targets old `gym`. **Do not fight this on native Windows.** All development happens on
  Linux from Phase 0 onward.
- **GPU** for MARL training (MAPPO/QMIX + CNN); a single modern GPU suffices for 32×32 grids but budget
  for many-seed sweeps.
- **Speed caveat (critical):** Cell2Fire runs as a **subprocess with CSV I/O per step**. This is slow and
  is the top practical risk to a many-seed MARL study. Profiling it is a Phase-1 go/no-go gate.
- **Frameworks:** `gymnasium`, `pettingzoo` (parallel API) for MARL, a maintained MARL library
  (`MARLlib`, `epymarl`, or RLlib multi-agent), `stable-baselines3` for single-agent baselines, `torch`.

---

## Phase dependency map

```
0 Cleanup/Migration ─► 1 Cell2Fire+Firehose integration ─► 2 Data ingestion (Saudi+California)
   └► 3 Critical infrastructure layer ─► 4 Single-agent baselines + heuristics (de-confound)
        └► 5 Decentralized MARL (Option B) ─► 6 Coordination & shared-reward cooperation
             └► 7 Hierarchical strategic prioritization (the contribution)
                  └► 8 GO/NO-GO: learned > heuristics?  (hard gate)
                       ├─ pass ─► 9 Cross-region transfer ─► 10 Ablations ─► 11 Validation/benchmark
                       │            └► 12 Statistical rigor ─► 13 Visualization ─► 14 Paper ─► 15 Certify
                       └─ fail ─► fallback: benchmark/negative-result workshop paper (documented)
```

Estimated total: **6–9 months** of focused work. Front-loaded risk in Phases 1, 5, and 8.

---

# Phase 0 — Cleanup, Archival & Migration Preparation
Estimated Time: 3–5 days · Compute: none (setup only).

## Objective
Freeze V1 as a citable historical artifact, carve out the reusable assets into a clean new package
structure, retire the toy simulator, and stand up a Linux development environment ready for Cell2Fire.
Leave the repo in a state where every subsequent phase has a clean surface to build on.

## Problems Addressed / Why it matters for AAAI
- V1's toy simulator and oracle heuristics cannot appear in a main-track paper; they must be clearly
  separated from the new work so reviewers (and you) never conflate them.
- A clean, documented migration preserves the reproducibility narrative ("we rebuilt on validated
  physics") instead of looking like an abandoned pivot.

## Files To Modify
| File | Change |
|------|--------|
| `legacy_v1/` (new dir) | Move the entire V1 custom-sim surface here: `src/wildfire_rl/envs/*`, `coordination/`, `routing/`, `ablation/`, V1 `scripts/*`, V1 `results/`, `figures/`. Tag it read-only. |
| `README.md` | Rewrite top-level: state the project is migrating to Cell2Fire; link `legacy_v1/` as the honest-negative-result prototype. |
| `pyproject.toml` | New package `wildfire_marl`; drop the SB3-only single-agent deps; add `gymnasium`, `pettingzoo`, MARL lib, `rasterio`/`gdal` (geospatial I/O). |
| `src/wildfire_marl/` (new package) | Empty skeleton: `env/`, `agents/`, `data/`, `infra/`, `eval/`, `viz/`, `experiments/`. |
| `src/wildfire_marl/eval/` | **Port, don't rewrite** from V1: `significance.py` (bootstrap CI, Cohen's d), `metrics.py` (ISR/WEL/CPS/RAC/PA/CCL/CE), `transfer.py` (TRS/CDGG/asymmetry). |
| `src/wildfire_marl/reproducibility/` | Port `logging_utils` (run manifests), `seeding`, `check_seed_integrity`, `make_manifest`, `validate_tensors`, `build_report_tables`. |
| `docs/MIGRATION.md` (new) | Record what carried over, what was retired, and *why* (the three AAAI rejection reasons). |
| `.github/workflows/ci.yml` | Keep lint/test/validate jobs; point them at `wildfire_marl`. |
| `environment-linux.yml` (new) | Conda env pinned for Linux incl. build deps (Eigen, gcc, make). |

## Step-by-Step Implementation Guide
1. **Branch & freeze.** Create branch `v2-cell2fire`; move V1 into `legacy_v1/` with a `FROZEN.md`
   stating it is the honest-negative-result prototype and is not part of the V2 contribution.
2. **Stand up Linux.** Provision WSL2/Linux/Colab; install build-essential, `cmake`, `make`, `gcc`,
   Eigen. Verify a trivial C++ program compiles.
3. **New package skeleton.** Create `src/wildfire_marl/` with the subpackages above; wire `pyproject`.
4. **Port the reusable core** (eval metrics, significance, reproducibility utilities) unchanged except
   import paths; keep their tests. These are your credibility ace — do not rewrite them.
5. **Preserve the data.** Keep `data/<region>/` rasters (NDVI/DEM/ERA5/FIRMS-derived tensors) — they
   feed Phase 2. Keep `docs/data_card.md` (update provenance).
6. **Document the migration** in `docs/MIGRATION.md` with the carries-over/retired table and rationale.

## Validation
```bash
# on Linux
python -c "import wildfire_marl; from wildfire_marl.eval import metrics, significance, transfer; print('core ported')"
pytest tests/  # ported metric/significance/transfer tests green
gcc --version && cmake --version   # build toolchain present
```

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] V1 fully relocated to `legacy_v1/` and frozen; new `wildfire_marl` package importable
- [ ] Ported eval metrics + significance + reproducibility utilities pass their tests on Linux
- [ ] Build toolchain (gcc/make/cmake/Eigen) verified

Reproducibility
- [ ] Region rasters + data card preserved and documented

Scientific validity / framing
- [ ] `docs/MIGRATION.md` states what carried over, what was retired, and the three rejection reasons it fixes

Platform
- [ ] Development environment is Linux/WSL/Colab (native-Windows path abandoned for the sim)

### Proceed Rule
- If ALL `[x]`, proceed to Phase 1. The rule for the whole V2 plan: **nothing from `legacy_v1/` is ever
  cited as a result** — it is prior art / motivation only.

---

# Phase 1 — Cell2Fire + Firehose Integration & Modernization
Estimated Time: 2–4 weeks · Compute: CPU build + light GPU. **Highest early integration risk.**

## Objective
Build Cell2Fire, get a Firehose-style single-agent suppression episode stepping end-to-end on stock
maps, modernize the interface to `gymnasium`, and **profile per-step speed** to size the whole project.

## Problems Addressed / Why it matters for AAAI
- Adopting a validated simulator is the credibility linchpin; it must actually run and be controllable.
- Speed determines how many seeds/ablations you can afford — decide feasibility now, not in month 4.

## Files To Modify
| File | Change |
|------|--------|
| `third_party/Cell2Fire/` (submodule) | Add Cell2Fire as a git submodule; build the C++ binary |
| `third_party/firehose/` (reference) | Vendor Firehose as a *reference fork you own* (it targets old `gym`, 2022) |
| `src/wildfire_marl/env/cell2fire_binding.py` (new) | Thin Python binding: launch the Cell2Fire subprocess, inject per-step harvest actions, parse the CSV state (modeled on Firehose `progress_to_next_state`) |
| `src/wildfire_marl/env/single_agent_env.py` (new) | Modern **Gymnasium** single-agent `FireSuppressionEnv`: obs = grid state channels, action = treat cell(s), reward = pluggable |
| `src/wildfire_marl/env/rewards.py` (new) | `FireSizeReward` (port) + hook for `InfrastructureWeightedReward` (Phase 3) |
| `scripts/profile_env.py` (new) | Time N steps × M episodes; extrapolate to a 5-seed study |

## Step-by-Step Implementation Guide
1. **Build Cell2Fire** (Eigen + `make` in `Cell2FireC/`); run the stock example; confirm it emits fire
   grids/perimeters.
2. **Revive Firehose enough to learn the interface**: run its `gym_env.py`/`evaluate_model.py --algo
   naive`; read how it injects actions and parses CSV state.
3. **Write a clean Gymnasium binding** (`cell2fire_binding.py` + `single_agent_env.py`) rather than
   depending on old-`gym` Firehose. Action space `Discrete(num_cells)` (treat one cell, `action_diameter`
   patch); observation `Box((C,H,W))` (fire/fuel/harvested channels). Do **not** modify Cell2Fire physics.
4. **Wire the reward hook** so any `Reward` subclass can be swapped in (Phase 3 needs this).
5. **Profile** (`profile_env.py`): measure seconds/step and seconds/episode; extrapolate to
   5 seeds × K episodes × the planned grids. Record the number.

## Validation
```bash
python scripts/profile_env.py --grid 32 --episodes 5
python -c "from wildfire_marl.env.single_agent_env import FireSuppressionEnv; e=FireSuppressionEnv(fire_map='Sub40x40'); o,_=e.reset(seed=0); e.step(e.action_space.sample()); print('env steps OK')"
```

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] Cell2Fire builds and runs the stock example on Linux
- [ ] Modern **Gymnasium** single-agent env steps end-to-end (reset/step/reward/render)
- [ ] Cell2Fire physics is **unmodified** (all suppression logic lives in the wrapper)

Reproducibility
- [ ] Deterministic given a seed (fixed ignition + fixed action sequence → identical trajectory)

Feasibility gate
- [ ] Per-step / per-episode time measured and recorded; a 5-seed study is affordable **or** a mitigation
      is chosen (smaller grid, patched IPC, parallel envs)

### Proceed Rule
- If the speed gate fails badly (e.g., hours per single run), **STOP and mitigate** before Phase 5 — a MARL
  study on a too-slow env is not viable. Options: reduce grid, vectorize/parallelize Cell2Fire processes,
  or patch the CSV IPC to shared memory.

---

# Phase 2 — Geospatial Data Ingestion (Saudi + California)
Estimated Time: 2–3 weeks · Compute: CPU.

## Objective
Convert your 7-channel Saudi/California tensors into **Cell2Fire landscape inputs** so the *validated
physics runs on your real data* — the thing neither PyroRL nor the V1 sim could do. Build both regions
with a **shared fuel model and shared encoding** so cross-region transfer is well-defined.

## Problems Addressed / Why it matters for AAAI
- "Your data is barely used" was a rejection reason. Here your NDVI/DEM/ERA5/FIRMS *drive the simulator*.
- Transfer is only meaningful if both regions share one fuel classification and encoding.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/data/to_cell2fire.py` (new) | Converter: 7-channel tensor → Cell2Fire landscape (`Forest.asc` fuel grid, `elevation.asc`, `Weather.csv`, fuel-lookup `Data.csv`, ignition list) |
| `src/wildfire_marl/data/fuel_mapping.py` (new) | NDVI → discrete **fuel-type codes** (Canadian FBP *or* Scott & Burgan). Documented, cited, single shared scheme for both regions |
| `data/cell2fire/Saudi/` , `data/cell2fire/California/` (new) | Generated landscapes (hashed into the manifest) |
| `src/wildfire_marl/data/ignition.py` (new) | FIRMS hotspots → ignition points; randomized-ignition sampler with disjoint train/eval seeds |
| `docs/data_card.md` | Update: fuel-model choice + mapping, CRS, shared normalization, ignition protocol |

## Step-by-Step Implementation Guide
1. **Choose ONE fuel model** (FBP or Scott–Burgan) and a defensible **NDVI→fuel-class** mapping; apply it
   identically to both regions. This is the one scientifically load-bearing mapping — cite sources.
2. **Elevation → slope/topography** in Cell2Fire's format from your DEM.
3. **ERA5 → `Weather.csv`** (wind speed/direction, temperature, humidity rows) at the sim's timestep.
4. **FIRMS → ignition points**; implement randomized ignition with `scenario_seed_offset` (disjoint
   train/eval) — port the V1 leakage-free protocol.
5. **Shared normalization/encoding** across regions (one scaler) so the observation space is common.
6. **Sanity-simulate**: run Cell2Fire on each region; confirm fires are physically plausible (spread with
   wind, self-extinguish in low fuel).

## Validation
```bash
python -m wildfire_marl.data.to_cell2fire --region saudi --grid 32
python -m wildfire_marl.data.to_cell2fire --region california --grid 32
python scripts/sim_smoke.py --region saudi   # a plausible fire perimeter + GIF
```

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] Both regions converted to valid Cell2Fire landscapes; the simulator runs on each
- [ ] Single shared fuel model + encoding + normalization across regions (documented)

Reproducibility
- [ ] Landscapes + fuel mapping hashed into the data manifest; converter is deterministic
- [ ] Randomized ignition with disjoint train/eval seeds (no leakage)

Scientific validity
- [ ] Simulated fires are physically plausible (wind-driven spread, low-fuel self-extinguish); a domain
      expert or the FBP tables sanity-check the fuel mapping

### Proceed Rule
- If the fuel mapping is arbitrary or region-specific, **fix it** — transfer conclusions are void otherwise.

---

# Phase 3 — Critical Infrastructure Layer (Petroleum, real-data-grounded)
Estimated Time: 2 weeks · Compute: CPU.

## Objective
Introduce **critical infrastructure that must be preserved with priority** — petroleum sites (Saudi) and
critical facilities/WUI (California) — grounded in **real GIS data**, as a value-weighted reward + an
observation channel, with catastrophe/cascade modeled **in the wrapper, never in Cell2Fire's physics**.

## Problems Addressed / Why it matters for AAAI
- V1's infrastructure parameters were invented — a rejection reason. Here they are real and cited.
- Protecting high-value assets (not just minimizing burned area) is the paper's domain contribution.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/infra/build_infrastructure.py` (new) | Rasterize **real** asset locations → `asset_type.asc`, `criticality.asc`, per-asset value; Saudi petroleum (public GIS), California critical facilities/WUI (public GIS) |
| `src/wildfire_marl/env/rewards.py` | `InfrastructureWeightedReward`: `−Σ value[asset]·fire_on_asset` on top of fire-size; `catastrophe_weight` for high-value assets |
| `src/wildfire_marl/env/single_agent_env.py`, `marl_env.py` | Add the **criticality observation channel**; track per-asset damage; expose `assets_reached`, `assets_detonated` in `info` |
| `src/wildfire_marl/infra/cascade.py` (new) | Optional cascade **in the wrapper**: a detonated asset seeds secondary ignitions near it (Cell2Fire physics untouched); records CCL |
| `docs/infra_card.md` (new) | Source, license, and provenance of every asset location + value |

## Step-by-Step Implementation Guide
1. **Acquire real asset data.** Saudi Eastern-Province petroleum facilities (public); California critical
   infrastructure / WUI communities (public GIS). Record source + license in `infra_card.md`.
2. **Value assignment** grounded in a defensible scheme (e.g., facility type / economic proxy), with a
   **sensitivity range** (Phase 8 ablates it) — no single hand-picked magic number decides a headline.
3. **Criticality raster** aligned to each landscape grid; add as an observation channel.
4. **`InfrastructureWeightedReward`** via the Phase-1 reward hook; the agent is now scored on
   *risk-weighted* protection, not just burned area.
5. **Cascade in the wrapper** (optional, off by default): a burning high-value asset seeds nearby
   ignitions in the *environment layer*; Cell2Fire's spread model stays pristine. Track CCL.
6. **Strategic metrics** (ISR/WEL/CPS/RAC/PCA/PA) re-anchored to Cell2Fire final state.

## Validation
```bash
python -m wildfire_marl.infra.build_infrastructure --region saudi
pytest tests/test_infra.py   # value-weighted reward orders refinery > pipeline > empty; obs channel added
```

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] Real asset rasters built for both regions; criticality observation channel added
- [ ] `InfrastructureWeightedReward` value-weights correctly (unit-tested)
- [ ] Cascade (if used) lives in the wrapper; **Cell2Fire physics unmodified**

Reproducibility
- [ ] Asset locations/values sourced, licensed, and provenance-documented (`infra_card.md`); hashed

Scientific validity
- [ ] No headline result depends on a single hand-picked asset value (sensitivity deferred to Phase 8)
- [ ] Strategic metrics (ISR/WEL/CPS/RAC/PA/CCL) compute on Cell2Fire final state

### Proceed Rule
- If asset values/locations are invented rather than sourced, **STOP** — this was a primary V1 rejection
  reason and must not recur.

---

# Phase 4 — Single-Agent Baselines & De-confounded Heuristics
Estimated Time: 2–3 weeks · Compute: GPU (PPO) + CPU (heuristics).

## Objective
Establish **fair, information-matched** baselines on the new env: heuristic cell-treatment policies (no
oracle privilege), a naïve PPO baseline (expected to struggle, per Firehose), and literature baselines.
This de-confounds every later comparison and reproduces the independent PPO-failure result.

## Problems Addressed / Why it matters for AAAI
- The **privilege confound** (heuristics reading true agent position) was a fatal V1 flaw. Fix it here.
- Reviewers demand baselines from the literature, not just noop/random.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/agents/heuristics.py` (new) | Information-matched heuristics: nearest-frontier treatment, greatest-risk-first, value-weighted-first — all using **only the observation**, no oracle state |
| `src/wildfire_marl/agents/ppo_baseline.py` (new) | Maskable-PPO + CNN (Firehose-style) single-agent |
| `src/wildfire_marl/eval/evaluate.py` (port) | Evaluation loop threading strategic metrics + disjoint eval seeds |
| `scripts/run_baselines.py` (new) | Heuristics + PPO + ≥1 literature baseline, both regions, bootstrap CIs |
| `docs/paper/related_baselines.md` (new) | Which literature methods reproduced and how |

## Step-by-Step Implementation Guide
1. **Information-matched heuristics**: every heuristic reads only the observation channels the RL policy
   sees. Document explicitly that no policy has oracle localization.
2. **PPO baseline** (Maskable-PPO + CNN); train on both regions with randomized ignition. Expect the
   Firehose difficulty (high-dim action space) — **record it honestly** as corroboration.
3. **≥1 literature baseline** (e.g., a firebreak-placement RL or Min/Max-L2 from Firehose) on your data.
4. **Effective-method gate** (port): the best reported method must beat no-op; report PPO's status
   honestly with effect sizes.

## Validation
```bash
python scripts/run_baselines.py --regions saudi california --seeds 5
python scripts/validate_learning_gate.py   # best method > noop; PPO status reported honestly
```

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] All policies are **information-matched** (no oracle privilege) — audited and documented
- [ ] Heuristic + PPO + ≥1 literature baseline evaluated on both regions with bootstrap CIs

Scientific validity
- [ ] Effective-method gate passes (a working method beats no-op)
- [ ] PPO's training difficulty reported honestly (corroborates Firehose's independent finding)
- [ ] Effect sizes + CIs on every comparison; no significance-without-effect

### Proceed Rule
- If any heuristic still reads privileged state, **STOP** — the whole comparison is invalid.

---

# Phase 5 — Decentralized Cooperating MARL (Option B)
Estimated Time: 4–8 weeks · Compute: GPU-heavy. **The core engineering lift.**

## Objective
Add an **agent layer** on top of Cell2Fire — **N firefighting agents, each with its own policy and local
observation**, moving over the grid and treating cells — and train them to **cooperate via a shared team
reward** using a MARL algorithm (**MAPPO / QMIX**) over the **PettingZoo** parallel API.

## Problems Addressed / Why it matters for AAAI
- V1's "MARL" was centralized single-policy; a stubbed high level. This delivers *genuine* decentralized
  cooperation — where the MARL-wildfire literature lives and where the ML novelty is.
- Cooperation + coordination (no two agents wasting effort on the same cell) is a real, measurable claim.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/env/marl_env.py` (new) | **PettingZoo `ParallelEnv`**: N agents with positions; per-agent **local observation** (egocentric crop of fire/fuel/criticality/other-agents) + global context; per-agent action = move (up/down/left/right/stay) + treat |
| `src/wildfire_marl/agents/agent_layer.py` (new) | Agent state (positions, movement, treat action) layered over the Cell2Fire binding; injects treated cells each step |
| `src/wildfire_marl/env/rewards.py` | **Shared team reward** = −(infrastructure-weighted burned) + coordination shaping (penalize redundant same-cell treatment) |
| `src/wildfire_marl/train/marl_train.py` (new) | MAPPO/QMIX training (MARLlib / epymarl / RLlib multi-agent); CTDE (centralized critic, decentralized actors) |
| `configs/marl/*.yaml` (new) | Agent count, local-view radius, algo hyperparameters |
| `src/wildfire_marl/eval/metrics.py` | Coordination Efficiency (CE) + redundant-dispatch metrics on the MARL rollout |

## Step-by-Step Implementation Guide
1. **PettingZoo ParallelEnv.** Define N agents; each observes an **egocentric local crop** (partial
   observability — the thing that makes MARL genuinely interesting) plus a coarse global channel; action =
   {move, treat}. Agents step **simultaneously**; the wrapper aggregates treated cells and advances
   Cell2Fire one step.
2. **Shared team reward** (cooperation): all agents share −(infrastructure-weighted damage). Add
   coordination shaping so two agents treating the same cell is wasteful (feeds CE).
3. **CTDE training** (MAPPO or QMIX): centralized critic during training, decentralized execution.
   Start MAPPO (robust for cooperative tasks); QMIX as a value-based comparison.
4. **Curriculum / action masking** to tame the high-dim action space (the Firehose failure mode): mask
   illegal/redundant treatments; optionally curriculum from small grids.
5. **Coordination metric (CE)** and redundant-dispatch tracking on rollouts; port the "no-coordination
   collapses" ablation design for Phase 10.

## Validation
```bash
python -m wildfire_marl.train.marl_train --config configs/marl/mappo_saudi.yaml --set total_steps=50000
python scripts/eval_marl.py --checkpoint <ckpt> --region saudi   # team contains fire; CE reported
```

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] PettingZoo ParallelEnv with N agents, **per-agent local observations**, simultaneous move+treat
- [ ] MAPPO **and** QMIX both train and run (CTDE); checkpoints saved
- [ ] Shared team reward + coordination shaping implemented

Scientific validity
- [ ] MARL team **cooperates** (CE high; agents cover distinct regions) — not N agents doing the same thing
- [ ] MARL beats the single best single-agent method on infrastructure-weighted metrics (CI-separated)

Reproducibility
- [ ] Deterministic seeds; run manifests + curves; seed integrity green

### Proceed Rule
- If MARL cannot beat the strong single-agent heuristic here, that is an early warning for the Phase-8
  gate — investigate reward/observation/coordination before adding the hierarchical layer.

---

# Phase 6 — Coordination & Cooperation Analysis
Estimated Time: 1–2 weeks · Compute: GPU.

## Objective
Turn "the agents cooperate" from an assertion into evidence: quantify emergent coordination, division of
labor, and the value of the shared reward vs. selfish rewards.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/eval/coordination.py` (new) | Metrics: CE, spatial division-of-labor, redundant-treatment rate, response latency to threatened assets (ERL) |
| `scripts/coordination_study.py` (new) | Shared-reward vs individual-reward; with vs without coordination shaping |
| `src/wildfire_marl/viz/coordination.py` (new) | Per-agent trajectory + sector-coverage overlays |

## Step-by-Step Implementation Guide
1. **Shared vs selfish reward** ablation: show shared team reward yields cooperation; selfish yields
   redundancy/collapse (your "no-coordination collapses" result, now on real physics).
2. **Division-of-labor** metric: agents cover distinct high-risk regions.
3. **Emergency response latency (ERL)** to threatened assets.

## Success Criteria (MANDATORY CHECKPOINT)
- [ ] Coordination is quantified (CE, division-of-labor, ERL) with CIs
- [ ] Shared reward measurably beats selfish reward on infrastructure protection (CI-separated)
- [ ] Coordination-shaping ablation shows its contribution

### Proceed Rule
- Proceed when cooperation is demonstrated with statistics, not anecdotes.

---

# Phase 7 — Hierarchical Strategic Prioritization (The Contribution)
Estimated Time: 3–5 weeks · Compute: GPU.

## Objective
Add a **learned strategic layer** that prioritizes *which critical assets/sectors the team defends*
under threat, over the decentralized low level — the paper's central method. Petroleum sites get
**priority attention** via learned, infrastructure-aware dispatch.

## Problems Addressed / Why it matters for AAAI
- V1's high level was a hand-coded stub. This is a *learned* strategic policy — the ML novelty.
- Infrastructure-aware prioritization is the domain contribution reviewers can point to.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/agents/strategic_controller.py` (new) | **Learned** high-level policy: obs = per-sector fire load + per-asset risk + team availability; action = per-agent target sector/asset (small, learnable action space) |
| `src/wildfire_marl/train/hierarchical_train.py` (new) | Two-timescale training: high-level RL over the small strategic space; low-level = trained MARL (Phase 5) executing dispatch |
| `configs/hierarchical/*.yaml` (new) | High-level algo, sectorization, priority weighting |
| `src/wildfire_marl/agents/heuristics.py` | Heuristic strategic baselines (greedy-risk, value-first) for the head-to-head |

## Step-by-Step Implementation Guide
1. **Small strategic action space** (which sector/asset each agent targets) — deliberately low-dim so it
   is *learnable* where flat cell-selection was not (the Firehose failure lesson).
2. **Two-timescale hierarchical training**: high level dispatches; low-level MARL executes.
3. **Head-to-head**: learned strategic vs heuristic strategic (greedy-risk / value-first) vs flat MARL vs
   PPO — all information-matched, on infrastructure metrics.
4. **Petroleum priority**: the strategic obs/reward weights high-value assets so the team defends them
   preferentially; measure PA/ISR on petroleum specifically.

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] Learned hierarchical policy trains (two-timescale); checkpoints saved
- [ ] Heuristic strategic baselines implemented for comparison

Scientific validity
- [ ] Petroleum/critical assets receive measurably higher protection (PA/ISR) under the strategic layer
- [ ] Learned strategic vs heuristic strategic quantified with CIs + effect sizes

### Proceed Rule
- Proceed to the Phase-8 gate with the full method assembled.

---

# Phase 8 — GO/NO-GO GATE: Does Learning Beat Heuristics?
Estimated Time: 2 weeks (analysis) · Compute: GPU (final training runs).

## Objective
The decisive scientific test. Determine, with statistical rigor, whether the **learned coordinated
hierarchical MARL** beats **strong heuristics** and **naïve PPO** on validated physics — the claim the
main-track paper rests on.

## Problems Addressed / Why it matters for AAAI
- Without a CI-separated win over heuristics, there is **no main-track contribution**. This gate prevents
  months of writing a paper that reviewers would reject.

## Step-by-Step Implementation Guide
1. **Final head-to-head** on both regions: {learned hierarchical MARL, flat MARL, best heuristic, PPO,
   noop} × infrastructure-weighted metrics, ≥5 seeds, bootstrap CIs + Cohen's d.
2. **Decision:**
   - **PASS** — learned method beats the best heuristic by a CI-separated margin on ≥1 primary metric,
     on ≥1 region (ideally both) → proceed to Phase 9 (main-track path).
   - **FAIL** — no CI-separated win → invoke the **documented fallback**: reframe as a *benchmark +
     honest negative-result* paper (validated env + fair baselines + reproducibility + the corroborated
     PPO-failure), targeting a **workshop / benchmark track**. Do not force a win.

## Success Criteria (MANDATORY CHECKPOINT)
- [ ] Full head-to-head with CIs + effect sizes on both regions
- [ ] A clear, pre-registered decision recorded (PASS → main track; FAIL → workshop fallback)
- [ ] No metric cherry-picking; primary metrics fixed **before** looking at results

### Proceed Rule
- **PASS → Phase 9. FAIL → workshop fallback.** Either way the work is publishable; only the *venue*
  differs. Never fabricate or relabel a win.

---

# Phase 9 — Cross-Region Transfer (Saudi ↔ California)
Estimated Time: 2–3 weeks · Compute: GPU (eval-heavy).

## Objective
Measure generalization **in both directions**: a policy **trained on Saudi, tested on California** (and
vice versa), quantifying **performance degradation**, transfer robustness, and adaptation asymmetry
between the desert-petroleum and forest-WUI regimes.

## Problems Addressed / Why it matters for AAAI
- Cross-domain generalization on validated physics with infrastructure metrics is a genuine transfer
  contribution (V1's was degenerate because the methods were region-agnostic; the *learned* policy here
  actually specializes, so the transfer signal is real).

## Files To Modify
| File | Change |
|------|--------|
| `configs/experiment/transfer.yaml` (new) | The symmetric matrix config (both regions, frozen-weight eval) |
| `scripts/run_transfer.py` (new) | Train-on-A / eval-on-B for all four cells; frozen weights; disjoint eval seeds |
| `src/wildfire_marl/eval/transfer.py` (port) | TRS / CDGG / adaptation asymmetry per metric |
| `src/wildfire_marl/viz/transfer.py` (new) | Degradation heatmaps + transfer-failure rollouts (native vs transferred side-by-side) |

## Step-by-Step Implementation Guide
1. **Fixed observation contract** across regions (same channels, same grid or a **fully-convolutional
   size-agnostic policy**) so frozen weights load on the other region.
2. **Shared scaler/encoding** (Phase 2) so the input space is common.
3. **Full matrix**: S→S, S→C, C→C, C→S with **frozen** weights and **held-out ignition seeds**.
4. **TRS/CDGG + adaptation asymmetry** per metric; **quantify degradation** (e.g., ISR drop Saudi→California).
5. **Qualitative transfer-failure rollouts** (native vs transferred) to make the domain shift legible.

## Success Criteria (MANDATORY CHECKPOINT)
Technical verification
- [ ] All four transfer cells computed with frozen weights + disjoint eval seeds
- [ ] Observation contract shared; policy loads cross-region

Scientific validity
- [ ] Performance degradation quantified in both directions (TRS/CDGG) with bootstrap CIs
- [ ] Adaptation asymmetry between desert-petroleum and forest-WUI regimes reported
- [ ] Transfer-failure rollouts rendered

### Proceed Rule
- Proceed when the symmetric matrix + asymmetry are complete with CIs.

---

# Phase 10 — Ablation Studies (AAAI-grade)
Estimated Time: 3–4 weeks · Compute: GPU.

## Objective
Isolate the causal contribution of each design choice with controlled, single-factor ablations on
validated physics.

## Ablation groups
1. **Learned hierarchical vs flat MARL vs heuristic vs PPO** (the architecture ablation).
2. **Shared vs selfish reward; coordination shaping on/off** (cooperation is necessary).
3. **Infrastructure-aware reward on/off; petroleum priority on/off** (the objective changes behavior).
4. **Local-view radius / partial observability** (observation sufficiency).
5. **Agent count** (1/3/5/10) — cooperative scaling on real physics.
6. **Infrastructure value / density sensitivity** (no headline depends on a magic number).
7. **Cascade on/off** (catastrophe modeling).

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/ablation/groups.py` (new) | Single-factor cells over the full proposed system |
| `scripts/run_ablations.py` (new) | Per-group tidy CSVs + bootstrap CIs + canonical rollout GIFs |
| `scripts/build_report_tables.py` (port) | Section-5 ablation tables from CSVs |

## Success Criteria (MANDATORY CHECKPOINT)
- [ ] Each ablation isolates one factor; deltas reported with CIs + effect sizes
- [ ] Each proposed component shown necessary (CI-separated degradation when removed) **or** honestly
      reported as not helping
- [ ] Infrastructure value/density **sensitivity** shows conclusions are parameter-robust
- [ ] Every cell → committed CSV + `run_id`; canonical rollout GIF per cell

### Proceed Rule
- Proceed when every architectural claim in the paper has an ablation behind it.

---

# Phase 11 — Environment Validation & Literature Benchmarking
Estimated Time: 2–3 weeks · Compute: CPU/GPU.

## Objective
Establish that the environment is credible (not a toy) and position results against prior work.

## Files To Modify
| File | Change |
|------|--------|
| `docs/paper/validation.md` (new) | Cell2Fire fire behavior vs real burn behavior / FBP expectations on a few events |
| `docs/paper/related_work.md` (new) | RL-for-wildfire, hierarchical/MARL, firebreak-RL, Firehose, transfer, safety-critical AI |
| `scripts/benchmark_baselines.py` (new) | Your method vs Firehose/firebreak-RL baselines on shared metrics |

## Success Criteria (MANDATORY CHECKPOINT)
- [ ] Environment fire behavior sanity-validated against real/expected behavior (a figure + discussion)
- [ ] ≥2 literature baselines reproduced and compared on shared metrics
- [ ] Related-work section situates the contribution honestly

### Proceed Rule
- Proceed when a reviewer could not dismiss the environment as a toy.

---

# Phase 12 — Statistical Rigor & Reproducibility (port + extend)
Estimated Time: 1–2 weeks · Compute: CPU.

## Objective
Re-assert the V1 reproducibility ace on the new stack: bootstrap CIs, effect-sizes-primary, verified
seed independence, manifests, provenance, CI gates.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/reproducibility/*` (port) | seeding, manifests, seed-integrity, provenance, tensor/landscape validation |
| `.github/workflows/ci.yml` | `validate` job: determinism + seed integrity + landscape validity + report↔CSV consistency + a CI-sized MARL smoke |
| `scripts/build_report_tables.py` (port) | All tables generated from CSVs; `source:`+SHA provenance |

## Success Criteria (MANDATORY CHECKPOINT)
- [ ] All reported numbers: mean + bootstrap 95% CI + Cohen's d; p never alone
- [ ] ≥5 verified-distinct seeds; seed-integrity green; no leakage (disjoint eval ignitions)
- [ ] Every table/figure → committed CSV + `run_id`; landscapes/models/assets hashed
- [ ] CI `validate` job blocking

### Proceed Rule
- Proceed when the reproducibility surface matches or exceeds V1 (which was already strong).

---

# Phase 13 — Visualization & Qualitative Analysis
Estimated Time: 1–2 weeks · Compute: CPU/GPU.

## Objective
Make behavior legible: cooperative agent coordination, infrastructure defense, cascade events, and
transfer failure — as figures and animations for the paper.

## Files To Modify
| File | Change |
|------|--------|
| `src/wildfire_marl/viz/rollout.py` (port/adapt) | Canonical animated rollout (fixed legend/palette) over Cell2Fire fire fields |
| `src/wildfire_marl/viz/strategic.py` (port) | Criticality overlay + per-agent dispatch + asset-defense trajectories |
| `scripts/render_all.py` (new) | Per (method × region × native/transfer) figures + GIFs |

## Success Criteria (MANDATORY CHECKPOINT)
- [ ] Canonical rollout GIF per method × region × {native, transfer}
- [ ] Cooperative coordination + infrastructure defense visually legible
- [ ] Transfer-failure rollouts (native vs transferred) rendered
- [ ] All figures generated from CSVs/checkpoints; none hand-drawn

### Proceed Rule
- Proceed when the qualitative story matches the quantitative one.

---

# Phase 14 — Paper Writing (AAAI Main Track Standards)
Estimated Time: 4–6 weeks · Compute: none.

## Objective
Write the paper to AAAI main-track standards: a clear contribution, fair baselines, validated env,
statistical rigor, honest limitations, and a reproducibility checklist.

## Deliverables
| File | Change |
|------|--------|
| `docs/paper/aaai_main.tex` (new) | Full paper: Intro · Related work · Problem formulation (risk-weighted multi-agent MDP + infrastructure) · Method (decentralized MARL + learned hierarchical prioritization) · Negative-result/failure analysis (PPO, corroborated) · Experiments (baselines, transfer, ablations, infrastructure) · Validation protocol · Limitations · Broader impact |
| `docs/paper/claims_evidence.md` (port) | Every claim → committed CSV / `run_id` / test |
| `docs/paper/checklist.md` (new) | AAAI reproducibility checklist, fully answered |
| `docs/paper/appendix.tex` (new) | Full ablation tables, transfer matrices, hyperparameters, compute, data/asset provenance |

## Success Criteria (MANDATORY CHECKPOINT)
Contribution & framing
- [ ] A single, crisp contribution statement; the learned-beats-heuristic result is central and CI-backed
- [ ] Negative-result/failure analysis is principled (reward-controllability / action-space argument), not anecdotal

Rigor
- [ ] Every quantitative claim maps to a committed artifact (claims→evidence map)
- [ ] Baselines are information-matched and include literature methods
- [ ] Limitations section is honest (env fidelity, fuel mapping, compute, single simulator)

Reproducibility
- [ ] AAAI reproducibility checklist fully answered; code+data release plan stated

### Proceed Rule
- Proceed to certification when the paper's every claim is traceable and defensible.

---

# Phase 15 — Reproducibility Certification & Submission Package
Estimated Time: 1–2 weeks + full-run compute.

## Objective
Certify end-to-end reproducibility from a clean clone, verify all artifact hashes, and assemble the
submission + supplementary + code/data release.

## Step-by-Step Implementation Guide
1. **Clean-clone dry run** on Linux: build Cell2Fire, install env from the lock, fetch models, verify
   hashes, run `make reproduce` (subset), confirm gates green + tests pass.
2. **Artifact evaluation package**: code, landscapes (or builders), models manifest, run logs, a
   one-command reproduction, and the reproducibility checklist.
3. **Camera-ready-quality figures/tables** regenerated from committed CSVs.

## Success Criteria (MANDATORY CHECKPOINT)
- [ ] Clean-clone build + reproduce + gates green + tests pass on a fresh machine
- [ ] All artifact hashes verified; certification log committed
- [ ] Submission + supplementary + code/data release prepared
- [ ] Every phase gate above is `[x]`

### Proceed Rule
- If ALL `[x]`, the paper is **submission-ready for the AAAI main track** (conditional on the Phase-8
  PASS). If Phase 8 was a FAIL, submit the certified **benchmark/negative-result** paper to the workshop
  track instead — do not overclaim.

---

# Honest Risk Register (read before starting)

| Risk | Phase | Severity | Mitigation |
|------|-------|----------|------------|
| Cell2Fire C++ build / Firehose old-`gym` rot | 1 | High | Linux/WSL/Colab; write a fresh Gymnasium wrapper rather than reviving old gym |
| Per-step subprocess/CSV speed too slow for MARL | 1,5 | High | Profile early (go/no-go); parallel envs, smaller grids, or patch IPC |
| **Learning does not beat heuristics** | 8 | **Critical** | Pre-registered go/no-go; documented workshop fallback; make the strategic problem genuinely hard (partial obs, coordination, high-dim) so learning has room |
| Fuel mapping (NDVI→fuel class) arbitrary | 2 | High | One shared, cited scheme; sanity-check vs FBP tables; sensitivity in Phase 10 |
| Infrastructure values invented (V1 relapse) | 3 | High | Real GIS sources + provenance + sensitivity analysis |
| MARL sample-inefficiency (high-dim actions) | 5,7 | High | Action masking, curriculum, and the *small* strategic action space at the high level |
| GPL-3.0 (Cell2Fire) release constraints | 15 | Low | Fine for academic release; document licensing |
| Single simulator (fidelity critique) | 11,14 | Medium | Validate against real behavior; state as a limitation; cite Cell2Fire's validation |

# Timeline summary

| Phases | Weeks | Milestone |
|--------|-------|-----------|
| 0–2 | 5–9 | Validated env runs on your real Saudi/California data |
| 3–4 | 4–5 | Infrastructure + fair baselines (de-confounded) |
| 5–7 | 8–15 | Decentralized cooperating MARL + learned hierarchical prioritization |
| **8** | 2 | **Go/no-go: learned > heuristics?** |
| 9–13 | 8–12 | Transfer, ablations, validation, rigor, visualization |
| 14–15 | 5–8 | Paper + certification |

**Total: ~6–9 months.** The project is AAAI-main-track-ready **iff** Phase 8 passes; otherwise it is a
strong, honest workshop/benchmark paper. Both are real outcomes — only the venue differs.
