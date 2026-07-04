"""Shared NDVI → Canadian-FBP fuel-type mapping (the scientifically load-bearing mapping).

**One scheme, both regions.** Cell2Fire's fire behavior is the Canadian Forest Fire Behavior
Prediction (FBP) System, so the fuel classification is FBP (not Scott & Burgan, which targets
Rothermel-based simulators). NDVI is mapped to FBP fuel types with fixed, region-independent
thresholds; the identical function is applied to Saudi and California physical NDVI, which is
what makes cross-region transfer well-defined (REMEDIATION_PLAN_V2, Phase 2).

Thresholds follow the standard remote-sensing interpretation of NDVI (barren < ~0.1; sparse
grass/shrub ~0.1–0.3; open woodland ~0.3–0.45; increasingly dense canopy above) and assign the
FBP fuel type of matching structure. References:

* Forestry Canada Fire Danger Group (1992). *Development and Structure of the Canadian Forest
  Fire Behavior Prediction System*. Information Report ST-X-3. (FBP fuel types C-1…C-7, O-1a/b.)
* Pais, C., Carrasco, J., Martell, D.L., Weintraub, A., Woodruff, D.L. (2021). *Cell2Fire: A
  Cell-Based Forest Fire Growth Model to Support Strategic Landscape Management Planning*.
  Frontiers in Forests and Global Change 4:692706. (Simulator + native FBP fuel inputs.)
* Tucker, C.J. (1979). *Red and photographic infrared linear combinations for monitoring
  vegetation*. Remote Sensing of Environment 8(2). (NDVI as green-biomass proxy.)
* USGS/NASA NDVI interpretation guidance: ~<0.1 barren rock/sand; ~0.2–0.5 sparse vegetation
  (grass/shrub); ~0.6+ dense green canopy.

The threshold values themselves are expert judgment over these sources (there is no canonical
NDVI→FBP table); they are therefore a **declared sensitivity parameter** — Phase 10 ablates
them, and no headline result may hinge on one bin edge.

Grid codes match the standard Cell2Fire/Prometheus ``fbp_lookup_table.csv``
(31=O-1a, 32=O-1b, 7=C-7, 5=C-5, 3=C-3, 101=Non-fuel).
"""

from __future__ import annotations

import numpy as np

# (upper NDVI bound, lookup grid code, Data.csv fueltype string, description)
# Bins are [prev_bound, bound); the last bin is open-ended.
NDVI_FBP_BINS: list[tuple[float, int, str, str]] = [
    (0.05, 101, "NF", "non-fuel: water / rock / sand / built"),
    (0.15, 31, "O1a", "O-1a matted grass: sparse arid vegetation"),
    (0.30, 32, "O1b", "O-1b standing grass: grassland / shrub-steppe"),
    (0.45, 7, "C7", "C-7 ponderosa pine / open woodland"),
    (0.60, 5, "C5", "C-5 red & white pine: moderate canopy"),
    (np.inf, 3, "C3", "C-3 mature jack/lodgepole pine: dense canopy"),
]

NONFUEL_CODE = 101

#: Grass fuel parameters written to Data.csv for O-1a/O-1b cells — Prometheus/Cell2Fire
#: conventions (grass fuel load kg/m^2 and degree of curing %), identical for both regions.
GRASS_FUEL_LOAD = 0.35
GRASS_CURING = 60.0

# ------------------------------------------------------------------ region NDVI recovery
#
# The committed V1 grids store a per-region "fuel density" channel, not NDVI. Physical NDVI is
# recovered with the *documented inverse* of each region's recorded V1 transform:
#
# * saudi:  V1 notebook 04_ndvi_pipeline_saudi applied  fuel = (NDVI + 1) / 2  (clip [0,1])
#           -> NDVI = 2*fuel - 1. Exact; cross-checked in-session against the raw MODIS tif
#           (data/saudi_eastern_province/raw/ndvi/, scale factor 1e-4).
# * california: V1 notebook 04_ndvi_pipeline_california applied a per-region min-max
#           normalization whose bounds were not recorded — NOT invertible. NDVI is
#           reconstructed with an ASSUMED affine range (below). This is a declared
#           approximation: raw MODIS NDVI for the California ROI must be re-downloaded to
#           close the gap (data-debt, tracked in docs/data_card.md), and Phase 10's fuel-map
#           sensitivity ablation covers it.

#: Assumed (min, max) physical NDVI over the California ROI in June, used to invert the
#: unrecorded V1 min-max: ocean/urban low end ~-0.05, dense conifer canopy high end ~0.90.
ASSUMED_CA_NDVI_RANGE: tuple[float, float] = (-0.05, 0.90)


def fuel_channel_to_ndvi(region: str, fuel: np.ndarray) -> np.ndarray:
    """Recover physical NDVI from a region's V1 fuel-density channel (see module notes)."""
    fuel = np.asarray(fuel, dtype=np.float64)
    if region == "saudi":
        return 2.0 * fuel - 1.0
    if region == "california":
        lo, hi = ASSUMED_CA_NDVI_RANGE
        return lo + (hi - lo) * fuel
    raise ValueError(f"Unknown region {region!r} (expected 'saudi' or 'california')")


# ------------------------------------------------------------------ the shared mapping


def ndvi_to_fbp_codes(ndvi: np.ndarray) -> np.ndarray:
    """Map physical NDVI to FBP lookup grid codes — the single shared scheme."""
    ndvi = np.asarray(ndvi, dtype=np.float64)
    codes = np.full(ndvi.shape, NONFUEL_CODE, dtype=np.int32)
    lower = -np.inf
    for upper, code, _name, _desc in NDVI_FBP_BINS:
        codes[(ndvi >= lower) & (ndvi < upper)] = code
        lower = upper
    return codes


def code_to_fueltype(code: int) -> str:
    """Lookup grid code -> Data.csv fueltype string (e.g. 31 -> 'O1a')."""
    for _upper, c, name, _desc in NDVI_FBP_BINS:
        if c == int(code):
            return name
    raise KeyError(f"Unknown FBP grid code {code}")


def class_distribution(codes: np.ndarray) -> dict[str, float]:
    """Fraction of cells per fueltype name (for the per-region sanity report)."""
    codes = np.asarray(codes)
    out: dict[str, float] = {}
    for _upper, code, name, _desc in NDVI_FBP_BINS:
        frac = float(np.mean(codes == code))
        if frac > 0:
            out[name] = round(frac, 4)
    return out
