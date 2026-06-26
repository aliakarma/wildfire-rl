"""Canonical PPO training pipeline (single source of truth).

Replaces the training loops duplicated in the Saudi/California trainers. Hyperparameters
come from :class:`PPOConfig`; the feature extractor is the single :class:`CustomCNN`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from wildfire_rl.config import PPOConfig
from wildfire_rl.logging_utils import get_logger
from wildfire_rl.models.cnn import CustomCNN
from wildfire_rl.seeding import set_global_seed

logger = get_logger(__name__)


def build_ppo(env, cfg: PPOConfig, seed: int = 0, tensorboard_log: str | None = None):
    """Construct a PPO model with the canonical CNN policy from ``cfg``."""
    from stable_baselines3 import PPO

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": cfg.features_dim, "use_pooling": cfg.cnn_pooling
        },
    }
    return PPO(
        cfg.policy,
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=cfg.verbose,
    )


def train_ppo(
    env_factory: Callable[[], object],
    cfg: PPOConfig,
    seed: int = 0,
    save_path: str | Path | None = None,
    tensorboard_log: str | None = None,
):
    """Seed, build a vec-env, train PPO for ``cfg.total_timesteps``, optionally save.

    Returns the trained model.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    set_global_seed(seed)
    vec_env = make_vec_env(env_factory, n_envs=cfg.n_envs, vec_env_cls=DummyVecEnv, seed=seed)
    model = build_ppo(vec_env, cfg, seed=seed, tensorboard_log=tensorboard_log)

    logger.info("Training PPO: %d timesteps, seed=%d", cfg.total_timesteps, seed)
    model.learn(total_timesteps=cfg.total_timesteps)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        logger.info("Saved model -> %s", save_path)
    return model
