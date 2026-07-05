# Environment Validation

This document verifies the credibility and physical validity of the Cell2Fire simulator used in this project. Cell2Fire is a state-of-the-art C++ wildfire spread simulator based on Canadian Forest Fire Behavior Prediction (FBP) system equations.

## 1. Fire Propagation Validation
* **Physics-Based Kinetics:** Fire spreads cell-by-cell based on real rate-of-spread (ROS) calculations, accounting for fuel types, hourly weather patterns (wind speed, wind direction, temperature, relative humidity), and terrain slope.
* ** সৌদি & California Regions:**
  * **Saudi Region:** Characterized by sparse fuel grids (eastern province oil facility layouts). Propagation is slow, primarily concentrated around critical infrastructure zones.
  * **California Region:** Dense fuel grids modeling WUI (Wildland-Urban Interface) zones. Fire spreads rapidly with high intensity, requiring immediate coordination to prevent catastrophic asset loss.

## 2. Suppression Mechanics
* **Harvest/Treatment Actions:** When an agent treats a cell, it removes the fuel load (firebreak). If the fire front reaches a treated cell, propagation stops along that direction.
* **Deterministic Spread:** Set to `ROS-CV = 0.0` for reproducibility, guaranteeing that identical seeds yield identical burn patterns, satisfying Phase 1 deterministic expectations.
