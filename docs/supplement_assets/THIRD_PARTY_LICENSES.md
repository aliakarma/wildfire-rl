# Third-party components and licenses

This archive contains our own code (MIT, see `LICENSE`) together with the third-party
components listed below. Each retains its original license.

## Cell2Fire (fire-spread simulator) — GPL-3.0

- **Location in this archive:** `third_party/firehose/cell2fire/`
- **License:** GNU General Public License v3.0 — see `third_party/firehose/LICENSE`
- **What it is:** a peer-reviewed cell-based fire-growth simulator implementing the Canadian
  Forest Fire Behaviour Prediction (FBP) system. It provides the fire physics for every
  experiment in the paper.
- **Upstream:** Cell2Fire, Pedro et al. The version vendored here is the variant patched by
  the Firehose RL benchmark to expose an interactive step protocol on stdin/stdout.
- **Our modifications to the simulator: none.** The physics is used unmodified. All
  suppression, agent, reward, and observation logic lives in our wrapper
  (`src/wildfire_marl/env/`). `cell2fire_binding.py` documents each deliberate difference
  from Firehose's process wrapper — all of them concern process control and seeding, not
  fire behaviour.
- **What is included:** C++ sources, headers, and `Makefile` only. Compiled objects
  (`*.o`), the precompiled header (`CellsFBP.h.gch`, 194 MB on its own), and the built
  binary are excluded under the 50 MB cap. Build instructions:
  `third_party/firehose/cell2fire/Cell2FireC/BUILD.md`.

## Firehose (RL benchmark harness) — see its own LICENSE

- **Location:** `third_party/firehose/` (Python helpers, stock benchmark landscapes under
  `third_party/firehose/data/`)
- Used as the reference for the interactive-simulator protocol and as a source of the stock
  20×20 / 40×40 benchmark maps that the environment smoke tests run against.

## Data sources

The raw geospatial inputs are **not** redistributed in this archive (see
`data/RAW_DATA_NOT_INCLUDED.md`). Their licenses, for reference:

| Source | Used for | Terms |
|---|---|---|
| NASA SRTM (3-arcsecond) | elevation, slope | Public domain (U.S. Government work) |
| Copernicus ERA5 (CDS) | wind, temperature, humidity | Copernicus Licence — free reuse with attribution |
| NASA FIRMS | active-fire hotspots, ignition candidates | Open, no restriction on reuse |
| Sentinel-2 derived NDVI | fuel proxy | Copernicus Licence — free reuse with attribution |

Critical-infrastructure locations are documented with per-record provenance in
`docs/infra_card.md`.

## GPL-3.0 note

Because Cell2Fire is GPL-3.0 and is distributed here in source form alongside our MIT-licensed
wrapper, redistribution of this archive as a whole is governed by the GPL-3.0 terms for the
Cell2Fire portion. Our own code remains independently available under MIT.
