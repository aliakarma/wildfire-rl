"""Modern Gymnasium single-agent fire-suppression environment over Cell2Fire.

A fresh implementation against the Gymnasium API (Firehose targets dead ``gym`` 0.21 and is
kept as reference only). One env instance owns one prepared input folder and one interactive
Cell2Fire subprocess (``Cell2FireBinding``).

* **Observation** — ``Box(0, 1, (3, H, W), float32)``: channel 0 = fire, channel 1 = harvested,
  channel 2 = fuel-available mask (static, from the FBP lookup). A (C, H, W) tensor so Phase 3
  can append the criticality channel and Phase 5 can crop per-agent egocentric views.
* **Action** — ``Discrete(num_cells)``: treat (harvest) one cell, expanded to an
  ``action_diameter``² patch like Firehose. ``action_masks()`` exposes valid-cell masking for
  Maskable-PPO (Phase 4) — the Firehose failure mode this plan targets is the huge flat action
  space, so masking support is wired in from day one.
* **Reward** — pluggable ``Reward`` subclass (``rewards.py``); default ``FireSizeReward``.
* **Determinism** — ``reset(seed=...)`` seeds ignition sampling (via ``self.np_random``) and the
  simulator (``--seed``); rate-of-spread noise is off by default (``ros_cv=0.0``).

Cell2Fire physics is never modified; everything here lives in the wrapper.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wildfire_marl.env.cell2fire_binding import (
    DEFAULT_BINARY,
    DEFAULT_DATA_DIR,
    Cell2FireBinding,
    read_grid_csv,
)
from wildfire_marl.env.rewards import FireSizeReward, Reward

_NODATA = -9999


def _read_asc_grid(path: Path) -> np.ndarray:
    """Read an ESRI ASCII grid (.asc), skipping the 6-line header."""
    return np.loadtxt(path, skiprows=6)


def _read_fbp_nonfuel_codes(lookup_csv: Path) -> set[int]:
    """Fuel-type codes that are non-fuel (cannot burn / cannot ignite).

    The lookup CSV columns are ``grid_value, export_value, descriptive_name, fuel_type, ...``;
    non-fuel rows (Non-fuel / Water / Unknown, grid codes 101/102/103) carry the *fuel_type*
    string ``Non-fuel``. Minimal parser — no dependency on the vendored Firehose package.
    (Phase-2 fix: the Phase-1 version matched a ``NF`` prefix against the descriptive-name
    column, which never matches — codes 101/102/103 were not excluded from the fuel mask.)
    """
    nonfuel: set[int] = {_NODATA}
    for line in lookup_csv.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].lstrip("-").isdigit():
            code, fuel_type = int(parts[0]), parts[3]
            if fuel_type.lower() in ("non-fuel", "nonfuel", "nf"):
                nonfuel.add(code)
    return nonfuel


class FireSuppressionEnv(gym.Env):
    """Single-agent cell-treatment suppression on validated Cell2Fire physics."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        fire_map: str = "Sub40x40",
        data_dir: str | Path = DEFAULT_DATA_DIR,
        binary: str | Path = DEFAULT_BINARY,
        max_steps: int = 200,
        steps_before_sim: int = 5,
        steps_per_action: int = 1,
        action_diameter: int = 1,
        reward_cls: type[Reward] = FireSizeReward,
        ignition_cell: int | None = None,
        ros_cv: float = 0.0,
        render_mode: str | None = None,
        verbose: bool = False,
        observe_infra: bool = False,
        catastrophe_weight: float = 0.0,
        cascade_prob: float = 0.0,
        infra_dir: str | Path | None = None,
        asset_values: dict[int, float] | None = None,
    ):
        """
        Args:
            fire_map: instance folder name under ``data_dir`` (e.g. ``Sub40x40``).
            data_dir: directory of Cell2Fire instances (default: vendored Firehose maps;
                Phase 2 points this at ``data/cell2fire/<region>``).
            binary: interactive Cell2Fire binary (Firehose-patched; physics unmodified).
            max_steps: truncation horizon (agent steps).
            steps_before_sim: fire periods simulated before the first action (fire head start).
                One fire period = 1 simulated MINUTE (``--Fire-Period-Length 1.0``; weather
                rows are hourly and advance every 60 periods).
            steps_per_action: fire periods advanced per agent action. 1 = Firehose-style
                minute-level control; 60 = one action per simulated hour (the region-episode
                default in configs — meaningful fire evolution between actions).
            action_diameter: 1 or 2 — treated patch size (1x1 or 2x2), as in Firehose.
            reward_cls: ``Reward`` subclass; instantiated with this env.
            ignition_cell: fixed 0-indexed ignition cell, or ``None`` to sample a random fuel
                cell per episode from ``self.np_random`` (seeded via ``reset(seed=...)``).
            ros_cv: Cell2Fire rate-of-spread coefficient of variation (0.0 = deterministic).
            verbose: log the subprocess protocol.
            observe_infra: add a normalized infrastructure-criticality observation channel.
            catastrophe_weight: reward penalty scale for fire on an asset (x asset value).
            cascade_prob: per-neighbor ignition prob when an asset cell burns.
            infra_dir: directory containing asset_type/criticality/blast_radius .npy files.
            asset_values: dict mapping asset codes to value multipliers.
        """
        super().__init__()
        self.map_dir = Path(data_dir) / fire_map
        if not self.map_dir.exists():
            raise FileNotFoundError(f"Unknown fire map: {self.map_dir}")
        self.fire_map = fire_map
        self.max_steps = int(max_steps)
        if action_diameter not in (1, 2):
            raise ValueError("action_diameter must be 1 or 2 (as in Firehose)")
        self.action_diameter = int(action_diameter)
        self.fixed_ignition_cell = ignition_cell
        self.render_mode = render_mode
        self.verbose = verbose

        # --- static landscape ---------------------------------------------------------
        self.forest = _read_asc_grid(self.map_dir / "Forest.asc")
        self.height, self.width = self.forest.shape
        self.num_cells = self.height * self.width
        nonfuel_codes = _read_fbp_nonfuel_codes(self.map_dir / "fbp_lookup_table.csv")
        self.fuel_mask = (~np.isin(self.forest, sorted(nonfuel_codes))).astype(np.float32)

        # --- infrastructure layers (Phase 3) -------------------------------------------
        self.observe_infra = observe_infra
        self.catastrophe_weight = float(catastrophe_weight)
        self.cascade_prob = float(cascade_prob)
        self.asset_values = asset_values or {1: 10.0, 2: 4.0, 3: 6.0, 4: 3.0}

        infra_path = Path(infra_dir) if infra_dir is not None else self.map_dir
        asset_type_path = infra_path / "asset_type.npy"
        criticality_path = infra_path / "criticality.npy"
        blast_radius_path = infra_path / "blast_radius.npy"

        if asset_type_path.exists():
            self.asset_type = np.load(asset_type_path).astype(np.int32)
        else:
            self.asset_type = None

        if criticality_path.exists():
            self.criticality = np.load(criticality_path).astype(np.float32)
        else:
            self.criticality = None

        if blast_radius_path.exists():
            self.blast_radius = np.load(blast_radius_path).astype(np.int32)
        else:
            self.blast_radius = None

        # --- spaces ---------------------------------------------------------------------
        self.action_space = spaces.Discrete(self.num_cells)
        obs_channels = 3
        self._active_observe_infra = self.observe_infra and self.criticality is not None
        if self._active_observe_infra:
            obs_channels = 4

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_channels, self.height, self.width), dtype=np.float32
        )

        # --- per-instance scratch input folder (Ignitions.csv rewritten per episode) ----
        self._input_dir = Path(tempfile.mkdtemp(prefix=f"c2f_in_{fire_map}_"))
        shutil.copytree(self.map_dir, self._input_dir, dirs_exist_ok=True)

        # --- simulator binding + reward -------------------------------------------------
        self.binding = Cell2FireBinding(
            input_folder=self._input_dir,
            binary=binary,
            steps_before_sim=steps_before_sim,
            steps_per_action=steps_per_action,
            ros_cv=ros_cv,
            verbose=verbose,
        )
        self.reward_func = reward_cls(self)

        # --- episode state ---------------------------------------------------------------
        self.fire_state = np.zeros((self.height, self.width), dtype=np.int8)
        self.ignition_cell: int | None = None
        self.iter = 0
        self.prev_actions: set[int] = set()

    # ------------------------------------------------------------------ helpers
    def _cells_in_patch(self, cell: int) -> list[int]:
        """0-indexed cells covered by an ``action_diameter``² patch anchored at ``cell``."""
        y, x = divmod(int(cell), self.width)
        if self.action_diameter == 1:
            coords = [(y, x)]
        else:  # 2x2, anchored top-left (Firehose convention)
            coords = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
        return [
            yy * self.width + xx
            for yy, xx in coords
            if 0 <= yy < self.height and 0 <= xx < self.width
        ]

    def _sample_ignition_cell(self) -> int:
        """Random fuel cell, drawn from the env's seeded RNG (leakage-free protocol)."""
        candidates = np.flatnonzero(self.fuel_mask.ravel() > 0)
        if candidates.size == 0:
            raise RuntimeError(f"No fuel cells to ignite on map {self.fire_map}")
        return int(self.np_random.choice(candidates))

    def _write_ignition_csv(self, cell_0indexed: int) -> None:
        # Cell2Fire is 1-indexed.
        (self._input_dir / "Ignitions.csv").write_text(f"Year,Ncell\n1,{cell_0indexed + 1}\n")

    def _refresh_state(self, csv_paths: list[Path]) -> None:
        if csv_paths:
            self.fire_state = read_grid_csv(csv_paths[-1]).astype(np.int8)

    def _obs(self) -> np.ndarray:
        channels = 4 if self._active_observe_infra else 3
        obs = np.zeros((channels, self.height, self.width), dtype=np.float32)
        obs[0] = self.fire_state > 0  # fire
        obs[1] = self.fire_state < 0  # harvested
        obs[2] = self.fuel_mask  # static fuel availability
        if self._active_observe_infra:
            obs[3] = self.criticality
        return obs

    def _info(self) -> dict[str, Any]:
        info = {
            "cells_on_fire": int(np.sum(self.fire_state > 0)),
            "cells_harvested": int(np.sum(self.fire_state < 0)),
            "ignition_cell": self.ignition_cell,
            "sim_finished": self.binding.finished,
        }
        if self.asset_type is not None:
            info["assets_reached"] = int(np.sum((self.asset_type > 0) & (self.fire_state > 0.1)))
            info["assets_detonated"] = len(self.previously_detonated)
            info["cascade_ignited"] = self.cascade_ignited_count
        else:
            info["assets_reached"] = 0
            info["assets_detonated"] = 0
            info["cascade_ignited"] = 0
        return info

    def action_masks(self) -> np.ndarray:
        """True where the action is still useful: fuel cell, not yet treated (for MaskablePPO)."""
        mask = self.fuel_mask.ravel() > 0
        if self.prev_actions:
            mask[np.fromiter(self.prev_actions, dtype=np.int64)] = False
        return mask

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        self.iter = 0
        self.prev_actions = set()
        self.fire_state = np.zeros((self.height, self.width), dtype=np.int8)
        self.previously_detonated = set()
        self.cascade_ignited_count = 0

        ignition = options.get("ignition_cell", self.fixed_ignition_cell)
        self.ignition_cell = int(ignition) if ignition is not None else self._sample_ignition_cell()
        self._write_ignition_csv(self.ignition_cell)

        # Simulator RNG seed: derive from the env seed stream so reset(seed=k) is total.
        sim_seed = int(self.np_random.integers(1, 2**31 - 1))
        csvs = self.binding.restart(seed=sim_seed)
        self._refresh_state(csvs)
        return self._obs(), self._info()

    def step(self, action):
        cell = int(action)
        if not 0 <= cell < self.num_cells:
            raise ValueError(f"Action {cell} outside Discrete({self.num_cells})")
        patch = self._cells_in_patch(cell)

        self.binding.apply_actions(patch)
        self.prev_actions.update(patch)
        csvs = self.binding.progress_to_next_state()
        self._refresh_state(csvs)

        # Post-spread wrapper cascade logic
        if self.cascade_prob > 0.0 and self.asset_type is not None:
            from wildfire_marl.infra.cascade import cascade_step

            self.fire_state, newly_det, newly_ign = cascade_step(
                fire_state=self.fire_state,
                asset_type=self.asset_type,
                blast_radius=self.blast_radius,
                fuel_mask=self.fuel_mask,
                previously_detonated=self.previously_detonated,
                cascade_prob=self.cascade_prob,
                rng=self.np_random,
            )
            self.previously_detonated.update(newly_det)
            self.cascade_ignited_count += newly_ign

        reward = float(self.reward_func(action=patch))
        self.iter += 1
        terminated = self.binding.finished
        truncated = (not terminated) and self.iter >= self.max_steps
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode != "rgb_array":
            raise NotImplementedError("Only render_mode='rgb_array' is supported")
        im = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        im[self.fuel_mask > 0] = (34, 139, 34)  # fuel: forest green
        im[self.fuel_mask == 0] = (120, 120, 120)  # non-fuel: grey
        im[self.fire_state > 0] = (255, 0, 0)  # fire: red
        im[self.fire_state < 0] = (28, 163, 236)  # harvested: blue
        if self.ignition_cell is not None:
            y, x = divmod(self.ignition_cell, self.width)
            im[y, x] = (255, 0, 255)  # ignition: pink
        return im

    def close(self):
        self.binding.close()
        shutil.rmtree(self._input_dir, ignore_errors=True)
        super().close()
