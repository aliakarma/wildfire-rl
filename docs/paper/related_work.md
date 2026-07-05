# Related Work

Situates this project in the context of state-of-the-art literature on reinforcement learning for wildfire suppression and multi-agent coordination.

## 1. RL for Wildfire Suppression
* Prior work (e.g., Firehose, firebreak-RL) uses simplified grid systems or toy cellular automata models.
* Our work extends these baselines by wrapping **Cell2Fire**, a C++ physics-based simulator, into a Gym/PettingZoo wrapper with actual Saudi/California regional data.

## 2. Multi-Agent Reinforcement Learning (MARL)
* Spatial coordination in cooperative tasks typically employs CTDE (Centralized Training with Decentralized Execution) algorithms like MAPPO or QMIX.
* Our study exposes the "coordination gap": pure decentralized MARL with local views struggles to protect global infrastructure assets without strategic direction.

## 3. Hierarchical MARL
* Hierarchical architectures split command logic: a slow-timescale manager selects goals, and fast-timescale workers execute actions.
* We implement a two-timescale learned high-level strategic director that dispatches low-level firefighting agents to threatened sectors.
