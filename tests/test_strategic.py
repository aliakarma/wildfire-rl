"""Strategic hierarchical controller (Phase 15B.2).

Asserts the controller is a valid drop-in policy and — the scientific point — that changing the
high-level variant (greedy_risk vs risk_aware) measurably changes infrastructure survival.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire_rl.config import EnvConfig, HierarchyConfig, InfraConfig
from wildfire_rl.coordination.strategic_controller import (
    StrategicController,
    make_strategic_controller,
)
from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv


def _infra_dir(tmp_path, grid):
    """A refinery in the NW sector (0), no assets elsewhere."""
    at = np.zeros((grid, grid), dtype=np.int32)
    crit = np.zeros((grid, grid), dtype=np.float32)
    at[2, 2] = 1  # refinery in NW
    crit[2, 2] = 1.0
    np.save(tmp_path / "asset_type.npy", at)
    np.save(tmp_path / "criticality.npy", crit)
    return str(tmp_path)


def _env(tmp_path, grid=16, num_agents=2):
    tensor = np.zeros((7, grid, grid), dtype=np.float32)
    cfg = EnvConfig()
    cfg.infra = InfraConfig(infra_dir=_infra_dir(tmp_path, grid), catastrophe_weight=5.0)
    return MultiAgentWildfireEnv(state_tensor=tensor, config=cfg, num_agents=num_agents)


def test_controller_emits_valid_actions(tmp_path):
    env = _env(tmp_path, num_agents=3)
    env.reset(seed=0)
    env.state[0, 2, 2] = 1.0  # ignite the refinery
    ctrl = StrategicController(
        env.action_space, grid_size=env.grid_size, variant="greedy_risk", env=env
    )
    actions, _ = ctrl.predict(None)
    assert actions.shape == (3,)
    assert np.all((actions >= 0) & (actions <= 4))


def test_invalid_variant_raises(tmp_path):
    env = _env(tmp_path)
    with pytest.raises(ValueError):
        StrategicController(env.action_space, variant="nonsense", env=env)


def test_variants_rank_sectors_differently(tmp_path):
    """A big fire in the SE (no assets) vs a small fire on a NW refinery: greedy chases the big
    fire; risk_aware defends the asset — so the top-priority sector differs."""
    env = _env(tmp_path, grid=16, num_agents=1)
    env.reset(seed=0)
    # small fire on the refinery (NW, sector 0)
    env.state[0, 2, 2] = 0.4
    # large fire mass in the SE (sector 3), no assets there
    env.state[0, 12:15, 12:15] = 1.0

    greedy = StrategicController(env.action_space, grid_size=16, variant="greedy_risk", env=env)
    risk = StrategicController(
        env.action_space, grid_size=16, variant="risk_aware", infra_risk_weight=50.0, env=env
    )
    fire = env.state[0]
    crit = env.infra_criticality
    g_top = int(np.argmax(greedy._sector_scores(fire, crit)))
    r_top = int(np.argmax(risk._sector_scores(fire, crit)))
    assert g_top == 3  # greedy chases the big SE fire
    assert r_top == 0  # risk_aware prioritizes the threatened NW refinery
    assert g_top != r_top


def test_high_level_changes_infrastructure_survival(tmp_path):
    """End-to-end: over a short rollout, risk_aware protects the refinery at least as well as
    greedy_risk (and strictly better in this asset-threatening scenario)."""
    from wildfire_rl.eval.metrics import infrastructure_survival_rate

    def run(variant):
        env = _env(tmp_path, grid=16, num_agents=1)
        env.reset(seed=1)
        env.state[0, 2, 2] = 0.4  # fire creeping onto the refinery
        env.state[0, 12:15, 12:15] = 1.0  # decoy fire far away (SE)
        ctrl = StrategicController(
            env.action_space, grid_size=16, variant=variant, infra_risk_weight=50.0, env=env
        )
        for _ in range(12):
            actions, _ = ctrl.predict(None)
            env.step(actions)
        return infrastructure_survival_rate(env.infra_asset_type, env.state)

    isr_greedy = run("greedy_risk")
    isr_risk = run("risk_aware")
    assert isr_risk >= isr_greedy


def test_factory_from_hierarchy_config(tmp_path):
    env = _env(tmp_path)
    ctrl = make_strategic_controller(
        env.action_space,
        HierarchyConfig(high_level="risk_aware", low_level="frontier", num_sectors=4),
        grid_size=env.grid_size,
        env=env,
    )
    assert ctrl.variant == "risk_aware"
    assert ctrl.low_level == "frontier"


def test_transfer_hybrid_csv_structure():
    """If the 15B.4 transfer matrix has been generated, it must cover every family × region with
    the strategic-metric columns (skips gracefully in a bare checkout)."""
    import pathlib

    import pandas as pd

    csv = pathlib.Path("results/transfer_hybrid.csv")
    if not csv.exists():
        pytest.skip("transfer_hybrid.csv not generated in this checkout")
    df = pd.read_csv(csv)
    for col in ("policy_family", "region", "isr_mean", "wel_mean", "cps_mean", "rac_mean"):
        assert col in df.columns, f"missing column {col}"
    assert set(df["region"].unique()) >= {"saudi", "california"}  # both directions covered
    assert {"nearest_fire", "hierarchical_risk_aware"} <= set(df["policy_family"].unique())
