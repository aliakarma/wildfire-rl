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
Final Saudi scaling experiments showed:
Metric	Value
1 Agent Mean	0.688
3 Agent Mean	0.629
5 Agent Mean	0.513
Interpretation:
•	increasing firefighter count improved wildfire containment by approximately 25%.
This validated:
•	cooperative suppression effectiveness.
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
to:
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
Metric	Value
Mean Episode Reward	-40525 ± 773
Mean Burned Cells	521 ± 6.75
Mean Fire Intensity	472 ± 5.73
Interpretation:
•	Saudi wildfire environments exhibited lower spread complexity,
•	lower fire intensity,
•	improved suppression effectiveness.
________________________________________
21. Final California PPO Results
Metric	Value
Mean Episode Reward	-86180 ± 1278
Mean Burned Cells	794 ± 3.20
Mean Fire Intensity	713 ± 2.50
Interpretation:
•	California wildfire regimes were substantially more difficult,
•	dense vegetation increased persistence,
•	higher fuel density amplified wildfire spread.
________________________________________
22. Cross-Regional Transfer Findings
The framework demonstrated substantial transfer degradation.
Key observation:
Policies trained under Saudi ecological conditions
do not generalize effectively to California wildfire regimes.
Performance degradation exceeded:
•	100% relative reward difference.
This strongly supports:
•	ecological specialization,
•	RL domain dependence,
•	environmental transfer limitations.
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
/content/drive/MyDrive/PyroRL_Saudi_Project/
Dataset directory:
/content/drive/MyDrive/PyroRL_Saudi_Project/datasets/
Tensor directories:
•	32x32,
•	64x64.
Model directories:
•	PPO checkpoints,
•	seed-specific models,
•	CSV result exports.
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

