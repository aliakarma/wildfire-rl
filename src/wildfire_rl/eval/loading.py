"""Robust PPO model loading helper with compatibility overrides.

Bypasses pickle/cloudpickle and NumPy version mismatches between
NumPy 1.x and 2.x, python version shifts, and stable-baselines3 shifts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from wildfire_rl.models.cnn import CustomCNN


def load_ppo_model(model_path: str | Path, env: Any) -> PPO:
    """Load a PPO checkpoint with custom_objects overrides for NumPy 1.x/2.x compatibility.

    Also handles size mismatches by trying both use_pooling=True and use_pooling=False.
    """
    # Base set of custom_objects to bypass pickle/cloudpickle errors
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 0.0003,
        "clip_range": lambda _: 0.2,
        "_last_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
    }

    # Try loading with cnn_pooling = True first
    try:
        custom_objects["policy_kwargs"] = {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "use_pooling": True,
            },
        }
        return PPO.load(str(model_path), custom_objects=custom_objects)
    except Exception as e:
        # If there's a size mismatch or key missing, try with use_pooling = False
        if "size mismatch" in str(e) or "Missing key" in str(e) or "Unexpected key" in str(e) or isinstance(e, RuntimeError):
            custom_objects["policy_kwargs"] = {
                "features_extractor_class": CustomCNN,
                "features_extractor_kwargs": {
                    "features_dim": 256,
                    "use_pooling": False,
                },
            }
            return PPO.load(str(model_path), custom_objects=custom_objects)
        raise e
