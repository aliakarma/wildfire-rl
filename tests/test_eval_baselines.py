"""Baseline policies + evaluation loop tests (torch-free)."""

from __future__ import annotations

from wildfire_rl.envs.base import make_env_factory
from wildfire_rl.eval.baselines import NoOpPolicy, RandomPolicy, NearestFirePolicy, FrontierPolicy
from wildfire_rl.eval.evaluate import evaluate_policy


def test_random_policy_evaluation(small_tensor, env_cfg):
    factory = make_env_factory(state_tensor=small_tensor, config=env_cfg)
    policy = RandomPolicy(factory().action_space, seed=0)
    result = evaluate_policy(policy, factory, n_episodes=3, base_seed=0, metrics_cfg=None)
    s = result["summary"]
    assert s["n_episodes"] == 3
    assert "episode_reward_mean" in s
    assert "burned_cells_mean" in s
    assert len(result["episodes"]) == 3


def test_noop_policy_runs(small_tensor, env_cfg):
    factory = make_env_factory(state_tensor=small_tensor, config=env_cfg)
    policy = NoOpPolicy(factory().action_space)
    result = evaluate_policy(policy, factory, n_episodes=2, base_seed=5)
    assert result["summary"]["n_episodes"] == 2


def test_heuristic_policies_run(small_tensor, env_cfg):
    factory = make_env_factory(state_tensor=small_tensor, config=env_cfg)
    env = factory()
    p1 = NearestFirePolicy(env.action_space, env=env)
    p2 = FrontierPolicy(env.action_space, env=env)
    
    r1 = evaluate_policy(p1, factory, n_episodes=2, base_seed=10)
    r2 = evaluate_policy(p2, factory, n_episodes=2, base_seed=10)
    
    assert r1["summary"]["n_episodes"] == 2
    assert r2["summary"]["n_episodes"] == 2


def test_privileged_state_flags():
    """Phase 10: the oracle-localization asymmetry must be self-declared and queryable."""
    assert NearestFirePolicy.uses_privileged_state is True
    assert FrontierPolicy.uses_privileged_state is True
    assert RandomPolicy.uses_privileged_state is False
    assert NoOpPolicy.uses_privileged_state is False


def test_evaluation_reproducible(small_tensor, env_cfg):
    factory = make_env_factory(state_tensor=small_tensor, config=env_cfg)
    p1 = RandomPolicy(factory().action_space, seed=0)
    p2 = RandomPolicy(factory().action_space, seed=0)
    r1 = evaluate_policy(p1, factory, n_episodes=3, base_seed=0)
    r2 = evaluate_policy(p2, factory, n_episodes=3, base_seed=0)
    assert r1["summary"]["episode_reward_mean"] == r2["summary"]["episode_reward_mean"]


def test_multi_agent_baselines(small_tensor, env_cfg):
    from wildfire_rl.envs.multi_agent import MultiAgentWildfireEnv
    
    def factory():
        return MultiAgentWildfireEnv(state_tensor=small_tensor, config=env_cfg, num_agents=3)
        
    env = factory()
    p1 = NearestFirePolicy(env.action_space, env=env)
    p2 = FrontierPolicy(env.action_space, env=env)
    
    r1 = evaluate_policy(p1, factory, n_episodes=2, base_seed=10)
    r2 = evaluate_policy(p2, factory, n_episodes=2, base_seed=10)
    
    assert r1["summary"]["n_episodes"] == 2
    assert r2["summary"]["n_episodes"] == 2
