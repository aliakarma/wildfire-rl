# Building the Cell2Fire simulator

The fire-spread simulator is a C++ program. This archive ships its **source only** — the
compiled binary, object files, and the precompiled header (`CellsFBP.h.gch`, 194 MB by
itself) are excluded under the 50 MB supplement cap.

You only need this for the **full reproduction path** (running the environment). Verifying
the paper's numbers from the frozen results (README steps 2–3) needs no simulator.

## Requirements

- Linux (Ubuntu 22.04 or similar; WSL2 works). **Native Windows is not supported.**
- `g++` with C++11 support
- Boost headers (`libboost-all-dev`)
- Eigen3 (`libeigen3-dev`)

```bash
sudo apt-get update
sudo apt-get install -y build-essential libboost-all-dev libeigen3-dev
```

## Build

```bash
cd third_party/firehose/cell2fire/Cell2FireC
make
```

This produces the binary `Cell2Fire` in that same directory, which is exactly where
`src/wildfire_marl/env/cell2fire_binding.py` expects it:

```python
DEFAULT_BINARY = repo_root() / "third_party" / "firehose" / "cell2fire" / "Cell2FireC" / "Cell2Fire"
```

The directory layout of this archive was kept identical to the development tree precisely so
that no path edits are needed after building.

If your Boost or Eigen headers live somewhere unusual, adjust the include paths at the top of
`Makefile`. The first compile takes a few minutes, mostly on the precompiled header.

## Verify the build

```bash
# from the archive root, with the Python env installed
python -c "
from wildfire_marl.env.cell2fire_binding import DEFAULT_BINARY
print('binary present:', DEFAULT_BINARY.exists())
"
python scripts/sim_smoke.py
```

`sim_smoke.py` runs a short episode on a stock benchmark landscape and prints the burned-cell
count. If it completes, the simulator is wired up correctly and you can proceed to the
evaluation commands in `docs/REPRODUCIBILITY.md`.

## Note on modifications

The simulator's fire physics is **unmodified**. The vendored copy is the Firehose-patched
variant, which adds an interactive stdin/stdout step protocol so an RL agent can act between
fire periods. All suppression, agent, reward, and observation logic lives in our wrapper under
`src/wildfire_marl/env/`. The deliberate differences from Firehose's own process wrapper
(seedable episodes, readline timeouts, configurable action cadence) are documented in the
module docstring of `cell2fire_binding.py`; none of them touch fire behaviour.
