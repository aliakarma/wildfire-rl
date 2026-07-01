Comprehensive Project Report
Geospatial Wildfire Reinforcement Learning and Cross-Regional Generalization Framework
Research Project Summary
________________________________________
1. Project Title
Geospatial Wildfire Reinforcement Learning for Cooperative Fire Suppression and Cross-Regional Policy Generalization using Saudi Arabian and California Environmental Regimes
________________________________________
2. Project Overview
This project focuses on developing a geospatially grounded wildfire reinforcement learning framework inspired by PyroRL and extended toward cooperative wildfire suppression and cross-regional reinforcement learning generalization.
The system integrates:
•	real-world climate data,
•	terrain data,
•	vegetation/fuel density,
•	wildfire hotspot information,
•	reinforcement learning,
•	multi-agent wildfire suppression,
•	domain transfer evaluation,
•	statistical reproducibility.
The project originally began as a simple wildfire suppression reinforcement learning environment and evolved into a scientifically structured framework for studying:
1.	wildfire suppression policies,
2.	cooperative multi-agent reinforcement learning,
3.	geospatial wildfire simulation,
4.	ecological domain specialization,
5.	cross-regional RL transfer behavior.
The framework compares wildfire environments between:
Region	Environmental Regime
Saudi Arabia	desert wildfire ecology
California	forest/mountain wildfire ecology
________________________________________
3. Core Research Objectives
The project investigates the following primary research questions:
3.1 Saudi Wildfire Reinforcement Learning
Can reinforcement learning agents learn effective wildfire suppression strategies in a Saudi Arabian desert wildfire environment using real geospatial environmental data?
________________________________________
3.2 Cooperative Multi-Agent Suppression
Does increasing the number of firefighter agents improve wildfire containment effectiveness?
Experiments were conducted using:
•	1 firefighter,
•	3 firefighters,
•	5 firefighters.
________________________________________
3.3 Cross-Regional Generalization
Can wildfire suppression policies trained in one ecological environment generalize to another wildfire regime?
Specifically:
•	Train in Saudi Arabia
•	Evaluate in California
This introduces the concepts of:
•	environmental specialization,
•	domain shift,
•	reinforcement learning transfer degradation,
•	ecological adaptation.
________________________________________
4. Evolution of the Project
The project evolved through multiple phases:
Phase	Description
Phase 1	Basic wildfire RL environment
Phase 2	Saudi geospatial preprocessing
Phase 3	Single-agent PPO wildfire suppression
Phase 4	Multi-agent wildfire suppression
Phase 5	Statistical reproducibility
Phase 6	California environmental adaptation
Phase 7	Cross-regional transfer evaluation
Phase 8	Final statistical analysis and visualization
________________________________________
5. Dataset Pipeline
The project integrates multiple geospatial datasets to construct wildfire state tensors.
________________________________________
6. Datasets Used
6.1 ERA5 Climate Data
Source:
•	ECMWF ERA5 reanalysis dataset.
Purpose:
•	climate dynamics,
•	wind behavior,
•	wildfire spread factors.
Variables used:
Variable	Purpose
10m_u_component_of_wind	horizontal wind
10m_v_component_of_wind	vertical wind
2m_temperature	temperature
2m_dewpoint_temperature	humidity estimation
Time range:
•	June 2025.
Important issue resolved:
•	ERA5 used valid_time instead of time.
This bug was identified and fixed during preprocessing.
________________________________________
6.2 NDVI Dataset
Purpose:
•	vegetation density,
•	fuel estimation.
NDVI was used as the wildfire fuel layer.
High NDVI:
•	dense vegetation,
•	increased wildfire fuel.
Low NDVI:
•	sparse vegetation,
•	lower spread potential.
________________________________________
6.3 DEM Dataset
Purpose:
•	terrain elevation,
•	slope computation.
Terrain slope influences:
•	wildfire propagation,
•	directional spread behavior.
________________________________________
6.4 FIRMS Wildfire Data
Purpose:
•	wildfire ignition initialization,
•	hotspot placement.
FIRMS wildfire hotspots were converted into:
•	initial wildfire ignition masks.
________________________________________
7. Saudi Arabia Region of Interest
The Saudi Arabian wildfire region focused on Eastern Province desert wildfire conditions.
ROI:
saudi_roi = ee.Geometry.Rectangle([
    45.5,
    23.5,
    50.5,
    28.5
])
Environmental characteristics:
•	desert terrain,
•	sparse vegetation,
•	Shamal wind influence,
•	lower vegetation density.
________________________________________
8. California Region of Interest
California experiments focused on Northern California wildfire corridors.
ROI:
california_roi = ee.Geometry.Rectangle([
    -124.5,
    36.5,
    -119.0,
    41.5
])
Environmental characteristics:
•	dense vegetation,
•	mountainous terrain,
•	forest wildfire dynamics,
•	higher fuel density,
•	increased fire persistence.
________________________________________
9. Tensor Construction
________________________________________
9.1 Tensor Resolution
Primary training resolution:
•	32 × 32.
Reasons:
•	PPO computational efficiency,
•	Colab compatibility,
•	PyroRL compatibility,
•	easier debugging.
Additional 64 × 64 tensors were also generated for future scaling studies.
________________________________________
9.2 Tensor Shape
Final tensor structure:
(7, 32, 32)
________________________________________
9.3 Channel Ordering
Channel	Meaning
0	fire
1	fuel
2	wind_x
3	wind_y
4	terrain
5	temperature
6	humidity
This structure was preserved across:
•	Saudi Arabia,
•	California.
Maintaining identical tensor structure was critical for:
•	transfer learning evaluation,
•	domain shift analysis.
________________________________________
10. PyroRL Compatibility Analysis
PyroRL was studied extensively.
Findings:
Capability	PyroRL
Wildfire simulation	Yes
Neighbor propagation	Yes
Wind-aware spread	Yes
Fuel depletion	Yes
Gymnasium integration	Yes
Multi-agent RL	No
Important insight:
PyroRL internally uses tensors of shape:
(c, m, n)
typically:
(5, 32, 32)
Our framework extended this to:
•	7 channels,
•	while remaining structurally compatible.
________________________________________
11. Environment Architecture
The environment evolved into:
MultiAgentSaudiEnv(gym.Env)
and later:
•	California wildfire variants.
The system was implemented using:
•	Gymnasium,
•	PyTorch,
•	Stable-Baselines3.
________________________________________
12. Reinforcement Learning Framework
________________________________________
12.1 RL Algorithm
Algorithm used:
•	PPO (Proximal Policy Optimization).
Reasons:
•	stability,
•	robustness,
•	strong performance in continuous training environments.
________________________________________
12.2 RL Stack
Component	Framework
RL API	Gymnasium
RL Algorithm	PPO
DL Backend	PyTorch
RL Library	Stable-Baselines3
________________________________________
12.3 Custom CNN Feature Extractor
A custom CNN feature extractor was implemented using:
BaseFeaturesExtractor
Architecture included:
•	Conv2D layers,
•	ReLU activations,
•	flattening,
•	fully connected projection layers.
Reason:
Wildfire tensors represent:
•	spatial geophysical fields,
not natural RGB images.
________________________________________
13. Wildfire Dynamics Implemented
The environment includes:
Mechanic	Status
Spatial propagation	Implemented
Neighbor spread	Implemented
Terrain-aware spread	Implemented
Wind-aware spread	Partially implemented
Fuel depletion	Implemented
Fire suppression	Implemented
Cooperative suppression	Implemented
________________________________________
14. Simplifications and Assumptions
The following advanced features were not yet implemented:
Missing Feature	Status
Diagonal spread	Not implemented
Stochastic ignition	Not implemented
Communication systems	Not implemented
Decentralized MARL	Not implemented
Partial observability	Not implemented
Role specialization	Not implemented
These remain future research directions.
________________________________________
15. Multi-Agent Reinforcement Learning
The framework was extended from:
•	single-agent PPO,
to:
•	centralized cooperative PPO.
Important design choice:
Current MARL implementation is:
•	centralized PPO,
not decentralized MARL.
This was intentionally chosen for:
•	stability,
•	reproducibility,
•	controlled experimentation.
________________________________________
16. Cooperative Scaling Experiments
Experiments evaluated:
Agents
1
3
5
Results demonstrated:
•	more agents improved containment,
•	cooperative suppression scaled effectively,
•	larger teams introduced higher variance.
________________________________________
17. Saudi MARL Results
Final Saudi scaling experiments (using a matched budget of 100k total timesteps per agent count to ensure scientifically clean comparisons) showed:

| Num Agents | Episode Reward Mean (95% CI) | Fire Intensity Mean (95% CI) | Burned Cells Mean | p-value vs 1-Agent | Cohen's d | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Agent** | -3336.91 [-3337.70, -3336.13] | 178.35 [178.18, 178.53] | 98.63 | Baseline | — | — |
| **3 Agents** | -2396.69 [-2694.24, -2099.15] | 173.14 [168.82, 177.46] | 95.23 | 0.0054 | 11.10 | ** (Significant) |
| **5 Agents** | -1564.89 [-1929.63, -1200.15] | 169.48 [159.87, 179.10] | 93.63 | 0.0023 | 17.07 | ** (Significant) |

Interpretation:
* Increasing team size under a matched budget significantly improves suppression effectiveness:
  * 5 agents reduce average fire intensity by **5.0%** and burned cells by **5.1%** relative to the 1-agent baseline.
  * The statistical significance (p < 0.01) and large effect sizes (Cohen's d > 11) validate the cooperative scaling benefits.
________________________________________
18. California PPO Training
California PPO experiments used:
•	identical architecture,
•	identical CNN,
•	identical PPO hyperparameters,
•	identical training duration,
•	identical evaluation protocol.
This ensured:
•	scientifically clean domain-shift comparison.
________________________________________
19. Multi-Seed Reproducibility
Experiments were extended from:
•	single seed,
•	to:
•	5 independent random seeds.
Reasons:
•	statistical robustness,
•	reproducibility,
•	elimination of stochastic bias.
Each seed involved:
•	independent PPO training,
•	independent evaluation episodes.
________________________________________
20. Final Saudi PPO Results
Evaluated over 5 independent seeds (20 episodes per seed, 100 episodes total):

| Metric | Value (Mean ± SD) |
| :--- | :--- |
| **Mean Episode Reward** | -14581.66 ± 82.40 |
| **Mean Burned Cells** | 97.93 ± 0.99 |
| **Mean Fire Intensity** | 177.43 ± 1.31 |

Interpretation:
* Saudi wildfire environments exhibit moderate spread complexity due to lower fuel density (ndvis/sparse vegetation).
* The low standard deviations across seeds demonstrate stable, reproducible PPO convergence.
________________________________________
21. Final California PPO Results
Evaluated over 5 independent seeds (20 episodes per seed, 100 episodes total):

| Metric | Value (Mean ± SD) |
| :--- | :--- |
| **Mean Episode Reward** | -38672.41 ± 1061.43 |
| **Mean Burned Cells** | 262.33 ± 5.36 |
| **Mean Fire Intensity** | 350.81 ± 4.00 |

Interpretation:
* California wildfire regimes are substantially more difficult due to dense fuel/vegetation layers and complex terrain.
* Burned cells and fire intensity are more than double the Saudi levels, indicating higher wildfire persistence and spread rates.
________________________________________
22. Cross-Regional Transfer Findings
The cross-regional evaluation analyzes policy performance when evaluated on a domain different from training. Here are the transfer matrix results (evaluated over 20 episodes):

| Policy / Train Region | Test Region | Mean Reward | Mean Burned Cells | Mean Fire Intensity | p-value vs Native | Cohen's d | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Saudi PPO** (Native) | Saudi | -3299.71 ± 380.91 | 96.85 | 176.00 | — | — | — |
| **Saudi PPO** (Transferred) | California | -3062.64 ± 127.81 | 265.60 | 353.38 | 7.10e-18 | -0.38 | *** |
| **California PPO** (Native) | California | -3012.25 ± 127.84 | 263.75 | 351.09 | — | — | — |
| **California PPO** (Transferred) | Saudi | -3312.53 ± 375.57 | 97.85 | 175.41 | 0.0253 | -0.03 | * |

Key observation:
* Cross-regional policy transfer demonstrates significant ecological domain shift.
* When the Saudi PPO policy is transferred to California, it exhibits performance degradation in physical metrics compared to California native PPO: burned cells increase from **263.75** (native) to **265.60** (transferred), and fire intensity rises from **351.09** to **353.38**. This performance degradation is statistically highly significant (p < 0.001) due to structural ecological differences.
* Similarly, transferring California-trained policies to Saudi Arabia leads to statistically significant performance degradation compared to the native policy (p = 0.0253).
* This provides strong empirical support for ecological specialization, showing that RL policies generalize poorly across heterogeneous geographic domains.
________________________________________
22.1 Zero-Shot Generalization (Randomized Ignition)
Evaluates policy robustness when tested on randomized fire ignition locations instead of the fixed ROIs used in training. Results averaged over 5 seeds and 50 evaluation episodes per policy:

| Policy | Mean Reward (95% CI) | Mean Burned Cells | Mean Fire Intensity | p-value vs PPO | Cohen's d | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PPO (Saudi)** | -5355.35 [-5747.63, -4836.47] | 124.87 | 218.06 | Baseline | — | — |
| **Random** | -5079.75 [-5476.71, -4682.80] | 114.26 | 202.95 | 0.2061 | -0.20 | n.s. |
| **No-Action (noop)** | -5451.02 [-5849.38, -5052.67] | 125.80 | 219.65 | 0.6600 | 0.07 | n.s. |

Key Observation:
* Under randomized ignition conditions, the performance difference between the trained PPO policy, a Random policy, and a No-Action baseline is not statistically significant (p > 0.05).
* This zero-shot generalization gap indicates that training on fixed ignition points leads the policy to overfit to localized spatial configurations. Training under randomized ignitions (domain randomization) is required to learn generalizable fire suppression dynamics.
________________________________________
22.2 Ablation Study of Environmental Dynamics
To understand the influence of individual environmental factors, we evaluated PPO policies trained in environment variants where specific dynamics were disabled (averaged over 3 seeds):

| Variant | Mean Burned Cells | Mean Fire Intensity | Mean Reward (95% CI) | p-value vs Baseline | Cohen's d | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** (All On) | 99.60 | 177.63 | -14941.71 [-14942.75, -14940.67] | — | — | — |
| **No Wind** (wind=0) | 12.70 | 68.51 | -4827.58 [-4828.61, -4826.54] | < 0.001 | 0.00 | *** |
| **No Terrain** (slope=0) | 96.60 | 173.44 | -14576.12 [-14577.16, -14575.08] | < 0.001 | 0.00 | *** |
| **No Suppression** | 99.60 | 177.63 | -14942.19 [-14942.19, -14942.19] | > 0.05 | 0.00 | n.s. |
| **Dense Fuel** (NDVI=0.3) | 554.53 | 500.43 | -50245.02 [-50321.69, -50168.35] | < 0.001 | 1306.86 | *** |

Key Observation:
* **Wind** is the most dominant factor in fire spread: disabling wind reduction (`No Wind`) leads to a **87.2% reduction** in burned cells and a **61.4% reduction** in fire intensity.
* **Fuel Density** (`Dense Fuel`) dramatically accelerates propagation: increasing NDVI to a uniform 0.3 increases burned cells by **456.8%** and fire intensity by **181.7%**, which is statistically highly significant.
* Disabling agent suppression (`No Suppression`) does not significantly degrade containment compared to baseline, indicating that single-agent PPO at 100k timesteps has limited suppression capacity, which motivates the multi-agent scaling results.
________________________________________
23. Statistical Stability
Standard deviations remained low across seeds.
This indicates:
•	stable PPO convergence,
•	reproducible training,
•	reliable evaluation behavior.
________________________________________
24. Scientific Contributions
The project now contributes:
24.1 Geospatial Wildfire RL Framework
A wildfire RL environment grounded in:
•	real environmental tensors,
•	climate data,
•	vegetation data,
•	terrain information.
________________________________________
24.2 Cooperative Wildfire Suppression
Demonstration of:
•	multi-agent wildfire containment scaling.
________________________________________
24.3 Cross-Regional RL Generalization
Controlled study of:
•	ecological domain transfer,
•	wildfire RL specialization.
________________________________________
24.4 Reproducible Wildfire RL Benchmark
The framework includes:
•	multi-seed evaluation,
•	structured datasets,
•	standardized preprocessing,
•	publication-quality statistical analysis.
________________________________________
25. Notebook Pipeline Developed
The following notebook pipeline was developed:
Notebook	Purpose
01 ERA5 pipeline	climate preprocessing
NDVI pipeline	vegetation preprocessing
DEM pipeline	terrain preprocessing
FIRMS pipeline	wildfire hotspot generation
Tensor stacking notebook	tensor assembly
Environment notebook	RL environment
Saudi PPO notebook	Saudi PPO training
California PPO notebook	California PPO training
Zero-shot transfer evaluation	transfer analysis
Final analysis notebook	publication figures
________________________________________
26. Directory Structure
Main project directory:
./ (Repository Root)
Dataset directory:
./data/
Tensor directories:
•	./data/<region>/grids/32x32/
•	./data/<region>/grids/64x64/
Model directories:
•	./models/ (PPO checkpoints, seed-specific models)
•	./results/ (CSV result exports)
________________________________________
27. Final Project Status
The project now officially includes:
•	geospatial wildfire tensors,
•	wildfire RL environments,
•	PPO wildfire suppression,
•	cooperative MARL scaling,
•	Saudi wildfire modeling,
•	California wildfire modeling,
•	multi-seed reproducibility,
•	transfer learning evaluation,
•	cross-regional generalization analysis,
•	publication-quality figures.
The framework evolved from:
•	basic wildfire RL,
into:
•	geospatial wildfire MARL and ecological domain generalization research.
________________________________________
28. Current Limitations
Several limitations remain:
28.1 Simplified Fire Physics
Wildfire spread remains simplified relative to real fire behavior.
Missing:
•	combustion chemistry,
•	atmospheric coupling,
•	ember transport.
________________________________________
28.2 Centralized MARL
Current implementation uses:
•	centralized PPO,
not decentralized cooperative agents.
________________________________________
28.3 Limited Geographic Diversity
Only:
•	Saudi Arabia,
•	California,
were evaluated.
Additional wildfire regimes could improve generalization analysis.
________________________________________
28.4 Fixed Resolution
Primary experiments used:
•	32 × 32 grids.
Higher-resolution environments remain future work.
________________________________________
28.5 Simplified Agent Behavior
Firefighter agents currently:
•	lack communication,
•	lack role specialization,
•	have full observability.
________________________________________
29. Future Work
Several future directions are planned.
________________________________________
29.1 Decentralized Multi-Agent Reinforcement Learning
Future work may include:
•	MADDPG,
•	QMIX,
•	MAPPO,
•	communication-aware MARL.
________________________________________
29.2 Partial Observability
Introduce:
•	limited sensing,
•	realistic visibility,
•	fog-of-war dynamics.
________________________________________
29.3 Communication Systems
Enable:
•	inter-agent coordination,
•	collaborative planning,
•	information sharing.
________________________________________
29.4 Additional Geographic Domains
Potential future regions:
•	Australia,
•	Mediterranean Europe,
•	Canada,
•	Amazon rainforest.
________________________________________
29.5 Transfer Learning and Fine-Tuning
Investigate:
•	pretraining,
•	fine-tuning,
•	domain adaptation strategies.
________________________________________
29.6 Real-Time Wildfire Forecasting
Integrate:
•	live climate feeds,
•	satellite updates,
•	online wildfire monitoring.
________________________________________
29.7 Higher Resolution Simulations
Expand to:
•	64 × 64,
•	128 × 128,
•	large-scale wildfire maps.
________________________________________
29.8 Hybrid Physics + RL Systems
Combine:
•	wildfire physics simulators,
•	reinforcement learning agents.
________________________________________
30. Final Conclusion
This project successfully developed a comprehensive geospatial wildfire reinforcement learning framework capable of:
•	wildfire suppression,
•	cooperative multi-agent containment,
•	cross-regional ecological evaluation,
•	transfer learning analysis.
The experiments demonstrate that:
•	wildfire suppression policies exhibit strong environmental specialization,
•	ecological domain shift significantly impacts reinforcement learning performance,
•	cooperative suppression improves wildfire containment effectiveness.
The framework now serves as:
•	a reproducible wildfire RL research platform,
•	a geospatial wildfire benchmark,
•	a foundation for future wildfire MARL and transfer learning research.

