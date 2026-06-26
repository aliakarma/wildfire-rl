"""Config loading + override tests."""

from __future__ import annotations

from wildfire_rl.config import Config, default_config, load_config, to_dict


def test_default_config_types():
    cfg = default_config()
    assert isinstance(cfg, Config)
    assert cfg.env.grid_size == 32
    assert cfg.metrics.burned_threshold == 0.5
    assert cfg.seeds == [0, 1, 2, 3, 4]


def test_dotlist_overrides():
    cfg = load_config(overrides=["ppo.total_timesteps=1000", "seed=7", "region.name=california"])
    assert cfg.ppo.total_timesteps == 1000
    assert cfg.seed == 7
    assert cfg.region.name == "california"


def test_to_dict_roundtrip_hashable():
    cfg = default_config()
    d = to_dict(cfg)
    assert isinstance(d, dict)
    assert d["env"]["grid_size"] == 32
