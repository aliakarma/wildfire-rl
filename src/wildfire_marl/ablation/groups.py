"""Ablation group definitions compiling results from multi-agent fire suppression runs.
"""

from __future__ import annotations


def get_ablation_records() -> list[dict[str, str | float]]:
    """Return consolidated ablation records across design configurations.

    Captures:
      - Architecture (Hierarchical vs Flat vs Heuristics)
      - Reward Penalties (Shared/Cooperative vs Selfish)
      - Agent count scaling (1 vs 3 agents)
      - Map regions (Saudi vs California)
    """
    return [
        # --- Saudi Arabia Region ---
        {
            "Region": "Saudi",
            "Ablation Group": "Architecture",
            "Configuration": "No-Op Baseline",
            "Return": -595.55,
            "Burned": 883.6,
            "WEL": 27.0,
            "CE": 1.00,
        },
        {
            "Region": "Saudi",
            "Ablation Group": "Architecture",
            "Configuration": "Value-First Heuristic",
            "Return": -594.50,
            "Burned": 882.1,
            "WEL": 21.4,
            "CE": 1.00,
        },
        {
            "Region": "Saudi",
            "Ablation Group": "Architecture",
            "Configuration": "Flat MARL (MAPPO)",
            "Return": -595.39,
            "Burned": 882.9,
            "WEL": 27.0,
            "CE": 0.97,
        },
        {
            "Region": "Saudi",
            "Ablation Group": "Architecture",
            "Configuration": "Learned Hierarchical",
            "Return": -595.27,
            "Burned": 883.0,
            "WEL": 27.0,
            "CE": 1.00,
        },
        {
            "Region": "Saudi",
            "Ablation Group": "Cooperation",
            "Configuration": "Shared Reward (Cooperative)",
            "Return": -615.25,
            "Burned": 892.8,
            "WEL": 27.0,
            "CE": 0.92,
        },
        {
            "Region": "Saudi",
            "Ablation Group": "Cooperation",
            "Configuration": "Selfish Reward (No Penalty)",
            "Return": -610.09,
            "Burned": 884.6,
            "WEL": 27.0,
            "CE": 0.76,
        },
        
        # --- California Region ---
        {
            "Region": "California",
            "Ablation Group": "Architecture",
            "Configuration": "No-Op Baseline",
            "Return": -379.96,
            "Burned": 658.4,
            "WEL": 24.1,
            "CE": 1.00,
        },
        {
            "Region": "California",
            "Ablation Group": "Architecture",
            "Configuration": "Value-First Heuristic",
            "Return": -378.42,
            "Burned": 655.2,
            "WEL": 23.3,
            "CE": 0.99,
        },
        {
            "Region": "California",
            "Ablation Group": "Architecture",
            "Configuration": "Flat MARL (MAPPO)",
            "Return": -379.96,
            "Burned": 658.4,
            "WEL": 24.1,
            "CE": 0.93,
        },
        {
            "Region": "California",
            "Ablation Group": "Architecture",
            "Configuration": "Learned Hierarchical",
            "Return": -379.96,
            "Burned": 658.4,
            "WEL": 24.1,
            "CE": 0.99,
        },
    ]
