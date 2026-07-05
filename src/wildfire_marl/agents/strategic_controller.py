"""Learned high-level strategic controller for hierarchical multi-agent fire suppression.

Divides the 32x32 grid into 16 sectors (4x4 grid of 8x8 patches) and assigns target sectors
to low-level agents.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StrategicController(nn.Module):
    """High-level Commander policy that dispatches N agents to 16 target sectors.

    Observation input:
      - sector_fire_load: [batch, 16] (sum of fire intensity per sector)
      - sector_asset_load: [batch, 16] (sum of criticality per sector)
      - agent_sectors: [batch, num_agents] (integer sector IDs for agent positions)
    Output:
      - Logits for MultiDiscrete([16, 16, 16]) actions (16 choices per agent).
    """

    def __init__(self, num_agents: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents

        # Input features: 16 fire loads + 16 asset loads + num_agents positions
        in_dim = 16 + 16 + num_agents
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Heads for each agent targeting one of 16 sectors
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 16) for _ in range(num_agents)])

    def forward(
        self,
        sector_fire: torch.Tensor,
        sector_asset: torch.Tensor,
        agent_sec: torch.Tensor,
    ) -> list[torch.Tensor]:
        # sector_fire: [B, 16]
        # sector_asset: [B, 16]
        # agent_sec: [B, num_agents]
        x = torch.cat([sector_fire, sector_asset, agent_sec], dim=-1)
        emb = self.encoder(x)
        logits_list = [head(emb) for head in self.heads]
        return logits_list


def get_sector_center(sector_idx: int, height: int = 32, width: int = 32) -> tuple[int, int]:
    """Helper to convert a sector index 0-15 to (y, x) grid coordinates."""
    grid_y = sector_idx // 4
    grid_x = sector_idx % 4
    sec_h = height // 4
    sec_w = width // 4
    # Center of the sector
    y = int(grid_y * sec_h + sec_h // 2)
    x = int(grid_x * sec_w + sec_w // 2)
    return y, x
