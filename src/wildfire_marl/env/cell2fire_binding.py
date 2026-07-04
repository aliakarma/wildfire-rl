"""Thin Python binding to the interactive Cell2Fire subprocess.

Drives the Firehose-patched Cell2Fire binary (``third_party/firehose/cell2fire/Cell2FireC``,
whose fire physics is verified unmodified — see ``third_party/README.md``) over its stdin/stdout
step protocol, modeled on Firehose's ``Cell2FireProcess.progress_to_next_state``:

  1. spawn the binary with ``--steps-before B --steps-action A --HarvestPlan``;
  2. the binary simulates; whenever it wants actions it prints ``Input action`` and blocks;
  3. we write one line of **1-indexed** cell ids to harvest (``0`` = no-op) to stdin;
  4. the binary advances ``A`` fire periods, printing the path of each per-period grid CSV
     (``ForestGrid*.csv``: 0 = untouched, 1 = burning/burned, -1 = harvested);
  5. ``Total Harvested Cells`` on stdout marks the end of the simulation.

Differences from Firehose's process wrapper (all deliberate, none touch the simulator):
  * ``--seed`` is a constructor argument (Firehose hardcodes 123) so episodes are seedable;
  * ``--ROS-CV`` is configurable and defaults to 0.0 (deterministic rate of spread) instead of
    0.5 — determinism is a Phase-1 gate; stochastic-spread studies can opt back in;
  * readline timeouts guard against a dead subprocess hanging training;
  * scratch I/O (input copy + output grids) lives in a ``tempfile`` directory (fast tmpfs on
    Linux) rather than inside the source tree.

All suppression/agent logic lives in the wrapper layer — Cell2Fire physics is never modified.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from wildfire_marl.paths import repo_root

#: Default location of the interactive (Firehose-patched) binary.
DEFAULT_BINARY = repo_root() / "third_party" / "firehose" / "cell2fire" / "Cell2FireC" / "Cell2Fire"
#: Default location of the stock benchmark maps (vendored Firehose data).
DEFAULT_DATA_DIR = repo_root() / "third_party" / "firehose" / "data"

_FINISHED_MARKER = "Total Harvested Cells"
_INPUT_MARKER = "Input action"


def _build_command(
    binary: Path,
    input_folder: Path,
    output_folder: Path,
    seed: int,
    steps_before_sim: int,
    steps_per_action: int,
    ros_cv: float,
    ignition_radius: int,
) -> list[str]:
    """Command line for the interactive binary (flag set modeled on Firehose)."""
    return [
        str(binary),
        "--input-instance-folder",
        f"{input_folder}{os.sep}",
        "--output-folder",
        f"{output_folder}{os.sep}",
        "--ignitions",
        "--sim-years",
        "1",
        "--nsims",
        "1",
        "--grids",
        "--final-grid",
        "--Fire-Period-Length",
        "1.0",
        "--output-messages",
        "--weather",
        "rows",
        "--nweathers",
        "1",
        "--ROS-CV",
        str(ros_cv),
        "--IgnitionRad",
        str(ignition_radius),
        "--seed",
        str(seed),
        "--nthreads",
        "1",
        "--ROS-Threshold",
        "0.1",
        "--HFI-Threshold",
        "0.1",
        "--steps-action",
        str(steps_per_action),
        "--steps-before",
        str(steps_before_sim),
        "--HarvestPlan",
    ]


class Cell2FireBinding:
    """One interactive Cell2Fire subprocess bound to one prepared input folder.

    The caller (the Gymnasium env) owns episode logic: it writes ``Ignitions.csv`` into
    ``input_folder`` before ``spawn()``/``restart()`` and converts grid CSVs to observations.
    """

    def __init__(
        self,
        input_folder: str | Path,
        binary: str | Path = DEFAULT_BINARY,
        seed: int = 0,
        steps_before_sim: int = 0,
        steps_per_action: int = 1,
        ros_cv: float = 0.0,
        ignition_radius: int = 0,
        read_timeout_s: float = 30.0,
        verbose: bool = False,
    ):
        self.binary = Path(binary)
        if not self.binary.exists():
            raise FileNotFoundError(
                f"Interactive Cell2Fire binary not found at {self.binary}. "
                "Build it first: cd third_party/firehose/cell2fire/Cell2FireC && "
                "make -f Makefile_UBUNTU EIGENDIR=/usr/include/eigen3/"
            )
        self.input_folder = Path(input_folder)
        self.seed = int(seed)
        self.steps_before_sim = int(steps_before_sim)
        self.steps_per_action = int(steps_per_action)
        self.ros_cv = float(ros_cv)
        self.ignition_radius = int(ignition_radius)
        self.read_timeout_s = float(read_timeout_s)
        self.verbose = verbose

        self._workdir = Path(tempfile.mkdtemp(prefix="c2f_out_"))
        self._spawn_count = 0
        self.process: subprocess.Popen | None = None
        self.lines: list[str] = []
        self.finished = False

    # ------------------------------------------------------------------ process
    @property
    def output_folder(self) -> Path:
        return self._workdir / f"run_{self._spawn_count}"

    def spawn(self) -> None:
        self.output_folder.mkdir(parents=True, exist_ok=True)
        cmd = _build_command(
            self.binary,
            self.input_folder,
            self.output_folder,
            seed=self.seed,
            steps_before_sim=self.steps_before_sim,
            steps_per_action=self.steps_per_action,
            ros_cv=self.ros_cv,
            ignition_radius=self.ignition_radius,
        )
        if self.verbose:
            print("Spawning:", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._spawn_count += 1
        self.finished = False
        self.lines = []

    def _read_line(self) -> str:
        """Read one stdout line (decoded, stripped) with a liveness guard."""
        assert self.process is not None, "spawn() first"
        line = self.process.stdout.readline().strip().decode("utf-8", errors="replace")
        self.lines.append(line)
        return line

    def progress_to_next_state(self) -> list[Path]:
        """Advance until the binary asks for input (or finishes); return new grid CSV paths."""
        assert self.process is not None, "spawn() first"
        csv_paths: list[Path] = []
        deadline = time.monotonic() + self.read_timeout_s
        line = None
        while line != _INPUT_MARKER:
            line = self._read_line()
            if line == "" and self.process.poll() is not None:
                break  # process exited; caller checks `finished`
            if line == "" and time.monotonic() > deadline:
                raise TimeoutError(
                    f"Cell2Fire produced no output for {self.read_timeout_s}s "
                    f"(last lines: {self.lines[-5:]})"
                )
            if self.verbose and line:
                print("[c2f]", line)
            if ".csv" in line and "Forest" in line and "We are plotting" not in line:
                csv_paths.append(Path(line))
            if _FINISHED_MARKER in line:
                self.finished = True
        return csv_paths

    def apply_actions(self, cells_0indexed: int | list[int] | None) -> None:
        """Send harvest cells for this step. ``None``/``-1`` = no-op (Cell2Fire id 0)."""
        assert self.process is not None, "spawn() first"
        if cells_0indexed is None:
            cells = [-1]
        elif isinstance(cells_0indexed, int):
            cells = [cells_0indexed]
        else:
            cells = list(cells_0indexed)
        # Cell2Fire cells are 1-indexed; our API is 0-indexed. -1 (no-op) maps to 0.
        payload = " ".join(str(c + 1) for c in cells) + "\n"
        self.process.stdin.write(payload.encode("utf-8"))
        self.process.stdin.flush()

    # ------------------------------------------------------------------ lifecycle
    def restart(self, seed: int | None = None) -> list[Path]:
        """Kill + respawn (new episode). Returns the grid CSVs from the pre-action burn-in."""
        if seed is not None:
            self.seed = int(seed)
        self.kill()
        # Drop the previous run's grids so disk/inodes don't grow across episodes.
        shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir.mkdir(parents=True, exist_ok=True)
        self.spawn()
        return self.progress_to_next_state()

    def kill(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.process.wait()
            self.process = None

    def close(self) -> None:
        self.kill()
        shutil.rmtree(self._workdir, ignore_errors=True)

    def __del__(self):  # best-effort cleanup
        import contextlib

        with contextlib.suppress(Exception):
            self.close()


def read_grid_csv(path: str | Path, timeout_s: float = 5.0):
    """Read a Cell2Fire grid CSV (rows of comma-separated ints) as a 2D numpy array.

    The binary writes CSVs asynchronously, so poll until the file is non-empty
    (ported from Firehose's ``wait_until_file_populated``).
    """
    import numpy as np

    p = Path(path)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if p.exists() and p.stat().st_size > 0:
            try:
                return np.loadtxt(p, delimiter=",", dtype=np.int8, ndmin=2)
            except ValueError:
                pass  # partially written; retry
        time.sleep(0.005)
    raise TimeoutError(f"Grid CSV never became readable: {p}")
