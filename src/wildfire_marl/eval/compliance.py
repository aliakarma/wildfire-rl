"""Compliance metrics to analyze target-following behavior of low-level agents."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def analyze_compliance_trajectory(
    positions_history: dict[str, list[tuple[int, int]]],
    targets_history: dict[str, list[tuple[int, int]]],
    dispatch_interval: int = 10,
) -> dict[str, Any]:
    """Computes target compliance, sector drift, and dispatch latency from histories.

    positions_history: {agent_name: list of (y, x)}
    targets_history: {agent_name: list of (y, x)}
    """
    agents = list(positions_history.keys())
    if not agents:
        return {}

    num_steps = len(positions_history[agents[0]])
    compliance_rates = []
    drifts = []
    latencies = []

    for agent in agents:
        pos = positions_history[agent]
        tar = targets_history[agent]

        # 1. Target Compliance Rate
        compliant_steps = 0
        for t in range(1, len(pos)):
            d_curr = compute_distance(pos[t], tar[t])
            d_prev = compute_distance(pos[t - 1], tar[t])
            # Compliant if we moved closer or are already at the target
            if d_curr < d_prev or d_curr <= 1.0:
                compliant_steps += 1
        comp_rate = compliant_steps / (len(pos) - 1) if len(pos) > 1 else 1.0
        compliance_rates.append(comp_rate)

        # 2. Sector Drift
        agent_drift = np.mean([compute_distance(pos[t], tar[t]) for t in range(len(pos))])
        drifts.append(float(agent_drift))

        # 3. Dispatch Latency
        # Measure how many steps it takes to reach a target sector (distance <= 2)
        # targets change every dispatch_interval steps
        lat_list = []
        for i in range(0, num_steps, dispatch_interval):
            window_pos = pos[i : i + dispatch_interval]
            window_tar = tar[i]
            reached = False
            for step_idx, p in enumerate(window_pos):
                if compute_distance(p, window_tar) <= 2.0:
                    lat_list.append(step_idx)
                    reached = True
                    break
            if not reached:
                lat_list.append(dispatch_interval)
        latencies.append(float(np.mean(lat_list)) if lat_list else float(dispatch_interval))

    # Sector Occupancy mapping (which sector of the 4x4 grid is the agent in)
    occupancy = dict.fromkeys(range(16), 0)
    total_visited = 0
    for agent in agents:
        for p in positions_history[agent]:
            sec_y = int(np.clip(p[0] // 8, 0, 3))
            sec_x = int(np.clip(p[1] // 8, 0, 3))
            sec_idx = sec_y * 4 + sec_x
            occupancy[sec_idx] += 1
            total_visited += 1

    sector_occupancy_ratio = {
        k: v / total_visited if total_visited > 0 else 0.0 for k, v in occupancy.items()
    }

    return {
        "target_compliance_rate": float(np.mean(compliance_rates)),
        "sector_drift": float(np.mean(drifts)),
        "dispatch_latency": float(np.mean(latencies)),
        "sector_occupancy": sector_occupancy_ratio,
    }
