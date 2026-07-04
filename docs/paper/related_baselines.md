# Related Baselines

This document outlines the literature baselines reproduced and benchmarked in Phase 4 of the project.

## 1. Non-Learned (Heuristic) Baselines

### 1.1 Frontier Heuristic (Firehose, 2024)
* **Description:** Selects fuel cells immediately adjacent to the active burning front. This standard perimeter-containment heuristic mimics real-world wildland firefighting, which focuses effort on the fire edge.
* **Reproduction in V2:** Implemented as `NearestFrontierPolicy`. It isolates cells at the burning edge using binary dilation and prioritizes cells closest to the active fire's centroid.
* **Citation:** *Firehose: Benchmarking Reinforcement Learning for Wildland Fire Suppression*, 2024.

### 1.2 Greatest Risk Heuristic
* **Description:** Focuses suppression on cells at the highest immediate risk of catching fire, defined by the local density of active burning neighbors.
* **Reproduction in V2:** Implemented as `GreatestRiskFirstPolicy`. It counts active burning neighbors in the 4-neighborhood of each treatable fuel cell and treats the cell with the highest burning neighbor count first.

### 1.3 Value-Weighted Frontier Heuristic
* **Description:** Incorporates spatial asset values (criticality maps) into perimeter suppression. Rather than treating all frontier cells equally, it targets the frontier cell defending the highest value asset.
* **Reproduction in V2:** Implemented as `ValueWeightedFirstPolicy`. It evaluates the criticality raster `obs[3]` at candidate frontier cells and prioritizes the one with the maximum criticality value (with risk/distance tie-breakers).

---

## 2. Learned (Reinforcement Learning) Baselines

### 2.1 Maskable PPO (sb3-contrib)
* **Description:** An on-policy actor-critic algorithm that natively supports action masking. Action masking is critical in high-dimensional discrete action spaces (such as 32x32 = 1024 cells) to prevent the agent from selecting invalid or redundant cells (e.g., non-fuel or already treated cells).
* **Reproduction in V2:** Implemented in `src/wildfire_marl/agents/ppo_baseline.py` using `sb3-contrib`. The environment's `action_masks()` output is injected via the `ActionMasker` wrapper, and the policy is trained using a small-grid-safe CNN feature extractor.
