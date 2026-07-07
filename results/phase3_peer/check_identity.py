"""Verify the H8 mechanism: ablation variants with identical eval rows have DIFFERENT
weights but IDENTICAL greedy decisions (inert RL stage + shared BC phase)."""

import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

import numpy as np
import torch

from wildfire_marl.agents.strategic_controller import StrategicController

TAGS = {
    "full_retrained": "checkpoint_ablation_full_system_retrained_saudi.pt",
    "wo_kl": "checkpoint_ablation_wo_kl_regularization_saudi.pt",
    "wo_entropy": "checkpoint_ablation_wo_entropy_bonus_saudi.pt",
    "bc_only": "checkpoint_ablation_bc_only_no_rl_fine_tuning_saudi.pt",
}

nets = {}
for name, fn in TAGS.items():
    ck = torch.load(Path("results/phase3_peer") / fn, map_location="cpu")
    net = StrategicController(num_agents=3)
    net.load_state_dict(ck["commander_state_dict"])
    net.eval()
    nets[name] = net

# 1. weight distance between variants
ref = nets["full_retrained"]
for name, net in nets.items():
    if name == "full_retrained":
        continue
    diffs = [
        (p1 - p2).abs().max().item()
        for p1, p2 in zip(ref.parameters(), net.parameters(), strict=True)
    ]
    print(f"max |w_full - w_{name}|: {max(diffs):.6f}")

# 2. greedy-decision agreement on 200 random sector states
rng = np.random.default_rng(0)
agree = {n: 0 for n in nets if n != "full_retrained"}
N = 200
for _ in range(N):
    s_fire = torch.tensor(rng.uniform(0, 30, 16), dtype=torch.float32).unsqueeze(0)
    s_asset = torch.tensor(rng.uniform(0, 10, 16), dtype=torch.float32).unsqueeze(0)
    s_agent = torch.tensor(rng.integers(0, 16, 3), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        ref_acts = [int(torch.argmax(x)) for x in ref(s_fire, s_asset, s_agent)]
        for name in agree:
            acts = [int(torch.argmax(x)) for x in nets[name](s_fire, s_asset, s_agent)]
            if acts == ref_acts:
                agree[name] += 1
for name, a in agree.items():
    print(f"greedy-decision agreement full vs {name}: {a}/{N} = {a / N:.1%}")
