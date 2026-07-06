"""Emergent coordination and division-of-labor evaluation metrics."""

from __future__ import annotations

import numpy as np


def spatial_division_of_labor(agent_paths: dict[str, list[tuple[int, int]]]) -> float:
    """DoL = Mean pairwise Jaccard distance between visited agent trajectories.

    1.0 = agents cover completely disjoint areas (perfect division of labor).
    0.0 = agents visit the exact same set of cells.
    """
    agents = list(agent_paths.keys())
    n = len(agents)
    if n < 2:
        return 1.0

    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            set_i = set(agent_paths[agents[i]])
            set_j = set(agent_paths[agents[j]])
            union = len(set_i.union(set_j))
            if union == 0:
                scores.append(1.0)
            else:
                intersection = len(set_i.intersection(set_j))
                scores.append(1.0 - (intersection / union))
    return float(np.mean(scores))


def redundant_treatment_rate(total_treatments: int, redundant_treatments: int) -> float:
    """RTR = Redundant treatments / Total treatments.

    Lower is better (0.0 = perfect suppression allocation).
    """
    if total_treatments == 0:
        return 0.0
    return float(redundant_treatments) / float(total_treatments)
