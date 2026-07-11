"""GIS-based wildfire rollout renderer for publication-quality figures.

Replaces the plain gray-grid renderer with a professional visualization that overlays
simulation results on real geographic basemap tiles (OpenStreetMap / Esri / OpenTopoMap).

The 32x32 simulation grid maps to real lat/lon coordinates (from ``Data.csv``), covering
a ~500 km regional extent.  Each cell is ~15-17 km — this is a **regional planning**
simulation, and the renderer honestly represents that scale.

Tile source selection & licensing
---------------------------------
1. **Esri World Topo** (``EsriWorldTopo``) — best print quality; attribution required:
   "Esri, HERE, Garmin, USGS, NGA, EPA, USDA" (free for non-commercial/academic).
2. **OpenTopoMap** (``OpenTopoMap``) — terrain contours, vegetation shading; ODbL license,
   attribution: "© OpenStreetMap contributors, SRTM | map style © OpenTopoMap (CC-BY-SA)".
3. **CartoDB Positron** (``CartoDBPositron``) — clean minimal style for supplementary
   figures; attribution: "© OpenStreetMap contributors © CARTO".

Usage::

    from wildfire_marl.viz.geo_renderer import GeoRenderer
    renderer = GeoRenderer("saudi", style="EsriWorldTopo")
    frame = renderer.render_frame(fire_state, asset_type, agent_positions, ...)

    # Presentation mode (1920x1080, larger fonts, stronger glow)
    renderer = GeoRenderer("saudi", theme="presentation")
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.path import Path as MPath
from PIL import Image

try:
    import contextily as cx
except ImportError:
    cx = None

# ---------------------------------------------------------------------------
# Tile providers
# ---------------------------------------------------------------------------

_TILE_PROVIDERS: dict[str, dict[str, Any]] = {
    "EsriWorldTopo": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri, HERE, Garmin, USGS, NGA, EPA, USDA",
        "name": "Esri World Topographic Map",
    },
    "OpenTopoMap": {
        "url": "https://tile.opentopomap.org/{z}/{y}/{x}.png",
        "attribution": "© OpenStreetMap contributors, SRTM | style © OpenTopoMap (CC-BY-SA)",
        "name": "OpenTopoMap",
    },
    "CartoDBPositron": {
        "url": "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png",
        "attribution": "© OpenStreetMap contributors © CARTO",
        "name": "CartoDB Positron",
    },
    "CartoDBDarkMatter": {
        "url": "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
        "attribution": "© OpenStreetMap contributors © CARTO",
        "name": "CartoDB Dark Matter",
    },
    "OSM": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "name": "OpenStreetMap",
    },
}

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

_THEMES: dict[str, dict[str, Any]] = {
    "paper": {
        "figsize": (8.0, 8.5),
        "dpi": 150,
        "bg": "#1a1a2e",
        "panel_bg": "#12122a",
        "panel_border": "#3a3a5c",
        "border_lw": 0.5,
        "fs": 1.0,
        "glow": 1.0,
        "fire_alpha": 0.72,
        "grid_alpha": 0.10,
        "wash_alpha": 0.06,
        "map_rect": [0.05, 0.13, 0.90, 0.77],
        "panel_rect": [0.05, 0.01, 0.56, 0.10],
        "legend_rect": [0.63, 0.01, 0.33, 0.10],
    },
    "presentation": {
        "figsize": (16.0, 9.0),
        "dpi": 120,
        "bg": "#0a0a1a",
        "panel_bg": "#0d0d20",
        "panel_border": "#2a2a4c",
        "border_lw": 0.3,
        "fs": 1.5,
        "glow": 2.0,
        "fire_alpha": 0.85,
        "grid_alpha": 0.06,
        "wash_alpha": 0.03,
        "map_rect": [0.03, 0.14, 0.94, 0.76],
        "panel_rect": [0.03, 0.01, 0.56, 0.11],
        "legend_rect": [0.61, 0.01, 0.36, 0.11],
    },
}

# ---------------------------------------------------------------------------
# Fire colormap
# ---------------------------------------------------------------------------

_FIRE_CMAP = LinearSegmentedColormap.from_list(
    "fire_age",
    ["#FFEB3B", "#FF9800", "#F44336", "#B71C1C", "#4A0000"],
    N=256,
)

# ---------------------------------------------------------------------------
# Agent colors / labels
# ---------------------------------------------------------------------------

_AGENT_COLORS = ["#00B0FF", "#00E676", "#E040FB"]
_AGENT_LABELS = ["A1", "A2", "A3"]

# ---------------------------------------------------------------------------
# Custom vector markers for assets (matplotlib Path objects)
# ---------------------------------------------------------------------------

_HOUSE_MKR = MPath(
    [(0, .55), (-.42, .0), (-.3, .0), (-.3, -.45), (.3, -.45), (.3, .0), (.42, .0), (0, .55)],
    [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO,
     MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY],
)

_DERRICK_MKR = MPath(
    [(0, .55), (-.28, -.45), (-.1, -.45), (-.04, .12), (.04, .12), (.1, -.45), (.28, -.45), (0, .55)],
    [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO,
     MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY],
)

# ---------------------------------------------------------------------------
# Region display metadata
# ---------------------------------------------------------------------------

_REGION_META: dict[str, dict[str, Any]] = {
    "saudi": {
        "display_name": "Saudi Arabia — Eastern Province",
        "asset_label": "Infrastructure",
        "asset_color": "#FF1744",
        "asset_marker": _DERRICK_MKR,
    },
    "california": {
        "display_name": "California — Northern Region",
        "asset_label": "WUI Community",
        "asset_color": "#FF6D00",
        "asset_marker": _HOUSE_MKR,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cell_coords(data_dir: str | Path, region: str) -> tuple[np.ndarray, np.ndarray]:
    """Read per-cell (lat, lon) from Data.csv. Returns arrays of shape (nrows, ncols)."""
    map_name = "Saudi" if region.lower() == "saudi" else "California"
    csv_path = Path(data_dir) / map_name / "Data.csv"
    lats, lons = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
    n = int(np.sqrt(len(lats)))
    return np.array(lats).reshape(n, n), np.array(lons).reshape(n, n)


def _lonlat_to_webmerc(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 lon/lat to Web Mercator (EPSG:3857) without pyproj."""
    x = lon * 20037508.342789244 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * 20037508.342789244 / np.pi
    return float(x), float(y)


def frame_interest_score(
    fire_state: np.ndarray, asset_type: np.ndarray | None,
) -> float:
    """Score how visually informative a frame is (for auto-snapshot selection).

    Balances activity (fire, suppression, defense) against basemap visibility.
    Heavily penalises frames where overlays cover >30% of the grid, which
    destroys the geographic context that makes the GIS renderer valuable.
    """
    n_burn = int((fire_state > 0).sum())
    n_sup = int((fire_state < 0).sum())
    n_total = max(fire_state.size, 1)

    score = min(n_burn, 30) * 2.0 + min(n_sup, 60) * 3.0
    if n_burn < 3 and n_sup < 2:
        score -= 15.0
    if asset_type is not None:
        score += int(((asset_type > 0) & (fire_state < 0)).sum()) * 5.0

    coverage = (n_burn + n_sup) / n_total
    if coverage > 0.25:
        score -= (coverage - 0.25) * 120.0
    return score


# =========================================================================
# GeoRenderer
# =========================================================================

class GeoRenderer:
    """GIS-aware rollout frame renderer with theme support.

    Parameters
    ----------
    region : str
        ``"saudi"`` or ``"california"``.
    style : str
        Tile provider key (see ``_TILE_PROVIDERS``).
    data_dir : str or Path
        Root Cell2Fire data directory.
    theme : str
        ``"paper"`` (print-optimised) or ``"presentation"`` (1920x1080 dashboard).
    dpi, figsize : overrides
        If provided, override the theme defaults.
    """

    def __init__(
        self,
        region: str,
        style: str = "EsriWorldTopo",
        data_dir: str | Path = "data/cell2fire",
        dpi: int | None = None,
        figsize: tuple[float, float] | None = None,
        cache_dir: str | Path | None = None,
        theme: str = "paper",
    ):
        self.region = region.lower()
        self._t = dict(_THEMES.get(theme, _THEMES["paper"]))
        self.dpi = dpi or self._t["dpi"]
        self.figsize = figsize or tuple(self._t["figsize"])
        self.meta = _REGION_META.get(self.region, _REGION_META["saudi"])

        if style not in _TILE_PROVIDERS:
            raise ValueError(f"Unknown style '{style}'; choose from {sorted(_TILE_PROVIDERS)}")
        self.tile_info = _TILE_PROVIDERS[style]
        self.style = style

        self.lats, self.lons = _load_cell_coords(data_dir, region)
        self.grid_size = self.lats.shape[0]

        lat_min, lat_max = float(self.lats.min()), float(self.lats.max())
        lon_min, lon_max = float(self.lons.min()), float(self.lons.max())
        dlat = (lat_max - lat_min) / (self.grid_size - 1) / 2
        dlon = (lon_max - lon_min) / (self.grid_size - 1) / 2
        self.lat_bounds = (lat_min - dlat, lat_max + dlat)
        self.lon_bounds = (lon_min - dlon, lon_max + dlon)

        x_min, y_min = _lonlat_to_webmerc(self.lon_bounds[0], self.lat_bounds[0])
        x_max, y_max = _lonlat_to_webmerc(self.lon_bounds[1], self.lat_bounds[1])
        self.extent_merc = (x_min, x_max, y_min, y_max)

        self._cell_x = np.zeros_like(self.lons)
        self._cell_y = np.zeros_like(self.lats)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                mx, my = _lonlat_to_webmerc(float(self.lons[r, c]), float(self.lats[r, c]))
                self._cell_x[r, c] = mx
                self._cell_y[r, c] = my

        cell_dx = abs(self._cell_x[0, 1] - self._cell_x[0, 0]) if self.grid_size > 1 else 1.0
        cell_dy = abs(self._cell_y[0, 0] - self._cell_y[1, 0]) if self.grid_size > 1 else 1.0
        self._cell_half_w = cell_dx / 2
        self._cell_half_h = cell_dy / 2

        self._basemap_img = None
        self._basemap_extent = None
        self._cache_dir = str(cache_dir) if cache_dir else None

        self._fire_history: dict[tuple[int, int], int] = {}
        self._step = -1

    # ------------------------------------------------------------------ basemap

    def _fetch_basemap(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if self._basemap_img is not None:
            return self._basemap_img, self._basemap_extent

        if cx is None:
            raise ImportError("contextily is required for GIS rendering. pip install contextily")

        kwargs: dict[str, Any] = {"source": self.tile_info["url"]}
        if self._cache_dir:
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

        img, ext = cx.bounds2img(
            self.extent_merc[0], self.extent_merc[2],
            self.extent_merc[1], self.extent_merc[3],
            zoom="auto", ll=False, **kwargs,
        )
        self._basemap_img = img
        self._basemap_extent = ext
        return img, ext

    # ------------------------------------------------------------------ geometry helpers

    def _cell_center(self, row: int, col: int) -> tuple[float, float]:
        return float(self._cell_x[row, col]), float(self._cell_y[row, col])

    def _rounded_patch(self, cx_: float, cy: float, expand: float = 1.0,
                       rpad_frac: float = 0.12) -> mpatches.FancyBboxPatch:
        rpad = self._cell_half_w * rpad_frac
        w = self._cell_half_w * 2 * expand
        h = self._cell_half_h * 2 * expand
        iw = max(w - 2 * rpad, rpad * 0.1)
        ih = max(h - 2 * rpad, rpad * 0.1)
        return mpatches.FancyBboxPatch(
            (cx_ - w / 2 + rpad, cy - h / 2 + rpad), iw, ih,
            boxstyle=f"round,pad={rpad}",
        )

    def _update_fire_history(self, fire_state: np.ndarray, step: int) -> None:
        if step <= self._step:
            self._fire_history.clear()
        self._step = step
        for r, c in np.argwhere(fire_state > 0):
            key = (int(r), int(c))
            if key not in self._fire_history:
                self._fire_history[key] = step

    # ================================================================== render

    def render_frame(
        self,
        fire_state: np.ndarray,
        asset_type: np.ndarray | None,
        agent_positions: dict[str, tuple[int, int]],
        strategic_targets: dict[str, tuple[int, int]],
        step_idx: int,
        wel: float,
        isr: float,
        ce: float,
        region: str,
        policy_name: str,
        burned_cells: int | None = None,
        suppressed_cells: int | None = None,
        prev_positions: list[dict[str, tuple[int, int]]] | None = None,
        comm_active: bool = False,
        save_pdf: str | Path | None = None,
    ) -> Image.Image:
        """Render one frame with GIS basemap + simulation overlay."""
        t = self._t
        fs = t["fs"]
        self._update_fire_history(fire_state, step_idx)

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, facecolor=t["bg"])
        ax = fig.add_axes(t["map_rect"])

        try:
            img, ext = self._fetch_basemap()
            ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear")
        except Exception:
            ax.set_facecolor("#2d2d3d")

        ax.set_xlim(self.extent_merc[0], self.extent_merc[1])
        ax.set_ylim(self.extent_merc[2], self.extent_merc[3])
        ax.set_aspect("equal")

        self._draw_basemap_wash(ax)
        self._draw_grid(ax)
        self._draw_fire(ax, fire_state, step_idx)
        self._draw_suppression(ax, fire_state)
        self._draw_assets(ax, asset_type, fire_state)
        if comm_active:
            self._draw_communication(ax, agent_positions, step_idx)
        self._draw_agents(ax, agent_positions, strategic_targets, prev_positions, step_idx)
        self._draw_title(ax, policy_name, fs)
        self._draw_scale_bar(ax, fs)
        self._draw_north_arrow(ax, fs)
        self._draw_attribution(ax)

        if burned_cells is None:
            burned_cells = int((fire_state > 0).sum())
        if suppressed_cells is None:
            suppressed_cells = int((fire_state < 0).sum())
        n_protected = 0
        total_assets = 0
        if asset_type is not None:
            n_protected = int(((asset_type > 0) & (fire_state <= 0)).sum())
            total_assets = int((asset_type > 0).sum())
        self._draw_metrics_panel(
            fig, step_idx, wel, isr, ce,
            burned_cells, suppressed_cells, n_protected, total_assets, fs,
        )
        self._draw_legend(fig, fs)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(t["panel_border"])
            spine.set_linewidth(t["border_lw"])

        if save_pdf:
            fig.savefig(
                str(save_pdf), format="pdf", dpi=self.dpi,
                facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight",
            )

        fig.canvas.draw()
        rgba = fig.canvas.buffer_rgba()
        pil = Image.frombytes("RGBA", fig.canvas.get_width_height(), rgba)
        plt.close(fig)
        return pil

    # ================================================================== layers

    def _draw_basemap_wash(self, ax) -> None:
        """Subtle dark wash to increase terrain contrast and push back map labels."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.fill_between([x0, x1], y0, y1, color="black", alpha=self._t["wash_alpha"], zorder=1)

    def _draw_grid(self, ax) -> None:
        """Grid lines via LineCollection at reduced opacity."""
        alpha = self._t["grid_alpha"]
        h_segs = []
        for r in range(self.grid_size + 1):
            yv = (self._cell_y[r, 0] + self._cell_half_h if r < self.grid_size
                  else self._cell_y[r - 1, 0] - self._cell_half_h)
            x0 = self._cell_x[0, 0] - self._cell_half_w
            x1 = self._cell_x[0, -1] + self._cell_half_w
            h_segs.append([(x0, yv), (x1, yv)])

        v_segs = []
        for c in range(self.grid_size + 1):
            xv = (self._cell_x[0, c] - self._cell_half_w if c < self.grid_size
                  else self._cell_x[0, c - 1] + self._cell_half_w)
            y0 = self._cell_y[0, 0] + self._cell_half_h
            y1 = self._cell_y[-1, 0] - self._cell_half_h
            v_segs.append([(xv, y0), (xv, y1)])

        lc = LineCollection(h_segs + v_segs, colors=(1, 1, 1, alpha), linewidths=0.3, zorder=2)
        ax.add_collection(lc)

    # ------------------------------------------------------------------ fire (organic)

    def _draw_fire(self, ax, fire_state: np.ndarray, step_idx: int) -> None:
        """Organic fire: rounded corners, soft glow, neighbouring merge."""
        burning = np.argwhere(fire_state > 0)
        if len(burning) == 0:
            return

        t = self._t
        max_age = max(step_idx, 1)
        glow_patches, glow_colors = [], []
        fire_patches, fire_colors, fire_ec = [], [], []
        hot_patches, hot_colors = [], []

        for r, c in burning:
            cx_, cy = self._cell_center(int(r), int(c))
            age = step_idx - self._fire_history.get((int(r), int(c)), step_idx)
            na = min(age / max(max_age * 0.5, 1), 1.0)
            rgba = list(_FIRE_CMAP(na))

            glow_patches.append(self._rounded_patch(cx_, cy, expand=1.35, rpad_frac=0.20))
            ga = (0.14 - 0.08 * na) * t["glow"]
            glow_colors.append([1.0, 0.82, 0.0, max(ga, 0.02)])

            fire_patches.append(self._rounded_patch(cx_, cy, expand=1.04, rpad_frac=0.14))
            rgba[3] = t["fire_alpha"] - 0.18 * na
            fire_colors.append(rgba)
            fire_ec.append([rgba[0], rgba[1], rgba[2], 0.40 - 0.20 * na])

            if age <= 2:
                hot_patches.append(self._rounded_patch(cx_, cy, expand=0.55, rpad_frac=0.25))
                hot_colors.append([1.0, 1.0, 0.85, 0.30 * (1.0 - age / 3.0)])

        ax.add_collection(PatchCollection(glow_patches, facecolors=glow_colors,
                                          edgecolors="none", zorder=3))
        ax.add_collection(PatchCollection(fire_patches, facecolors=fire_colors,
                                          edgecolors=fire_ec, linewidths=0.5, zorder=5))
        if hot_patches:
            ax.add_collection(PatchCollection(hot_patches, facecolors=hot_colors,
                                              edgecolors="none", zorder=6))

    # ------------------------------------------------------------------ firebreaks

    def _draw_suppression(self, ax, fire_state: np.ndarray) -> None:
        """Firebreaks: lighter cyan, rounded corners, diagonal hatch, faint glow."""
        treated = np.argwhere(fire_state < 0)
        if len(treated) == 0:
            return

        glow_p, fill_p, hatch_segs = [], [], []
        for r, c in treated:
            cx_, cy = self._cell_center(int(r), int(c))
            glow_p.append(self._rounded_patch(cx_, cy, expand=1.20, rpad_frac=0.18))
            fill_p.append(self._rounded_patch(cx_, cy, expand=1.0, rpad_frac=0.12))

            x0 = cx_ - self._cell_half_w
            y0 = cy - self._cell_half_h
            w = self._cell_half_w * 2
            h = self._cell_half_h * 2
            for i in range(1, 5):
                t_ = i / 5
                hatch_segs.append([(x0 + w * t_, y0), (x0, y0 + h * t_)])
                hatch_segs.append([(x0 + w, y0 + h * (1 - t_)), (x0 + w * (1 - t_), y0 + h)])

        ax.add_collection(PatchCollection(glow_p, facecolors=(0.3, 0.75, 1.0, 0.08),
                                          edgecolors="none", zorder=4))
        ax.add_collection(PatchCollection(fill_p, facecolors=(0.55, 0.92, 1.0, 0.28),
                                          edgecolors=(0.55, 0.95, 1.0, 0.75),
                                          linewidths=0.7, zorder=5))
        if hatch_segs:
            ax.add_collection(LineCollection(hatch_segs, colors=(0.55, 0.95, 1.0, 0.40),
                                             linewidths=0.4, zorder=6))

    # ------------------------------------------------------------------ assets (vector icons)

    def _draw_assets(self, ax, asset_type: np.ndarray | None, fire_state: np.ndarray) -> None:
        """Monochrome vector icons with colored outlines."""
        if asset_type is None:
            return
        asset_locs = np.argwhere(asset_type > 0)
        if len(asset_locs) == 0:
            return

        color = self.meta["asset_color"]
        marker = self.meta["asset_marker"]
        for r, c in asset_locs:
            cx_, cy = self._cell_center(int(r), int(c))
            burned = fire_state[r, c] > 0
            defended = fire_state[r, c] < 0

            if burned:
                fc, ec, alpha = "#666666", "#FF0000", 0.5
            elif defended:
                fc, ec, alpha = "#FFFFFF", "#00E676", 0.95
                ax.add_patch(mpatches.Circle(
                    (cx_, cy), self._cell_half_w * 0.75,
                    facecolor="none", edgecolor="#00E67650",
                    linewidth=1.8, zorder=7,
                ))
            else:
                fc, ec, alpha = "#FFFFFF", color, 0.95

            ax.scatter(cx_, cy, marker=marker, s=120, c=fc, edgecolors=ec,
                       linewidths=1.4, alpha=alpha, zorder=8)

    # ------------------------------------------------------------------ communication

    def _draw_communication(self, ax, agent_positions: dict[str, tuple[int, int]],
                            step_idx: int) -> None:
        """Bezier curves with animated pulse travelling along the link."""
        agents = sorted(agent_positions.keys())
        if len(agents) < 2:
            return

        segs, seg_colors = [], []
        n_pts = 24
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                r1, c1 = agent_positions[agents[i]]
                r2, c2 = agent_positions[agents[j]]
                x1, y1 = self._cell_center(r1, c1)
                x2, y2 = self._cell_center(r2, c2)
                dx, dy = x2 - x1, y2 - y1
                mx, my = (x1 + x2) / 2 - dy * 0.10, (y1 + y2) / 2 + dx * 0.10

                pulse_pos = (step_idx * 0.18) % 1.0
                pts = []
                for k in range(n_pts + 1):
                    t_ = k / n_pts
                    bx = (1 - t_) ** 2 * x1 + 2 * (1 - t_) * t_ * mx + t_ ** 2 * x2
                    by = (1 - t_) ** 2 * y1 + 2 * (1 - t_) * t_ * my + t_ ** 2 * y2
                    pts.append((bx, by))

                for k in range(len(pts) - 1):
                    t_mid = (k + 0.5) / n_pts
                    dist = abs(t_mid - pulse_pos)
                    if dist > 0.5:
                        dist = 1.0 - dist
                    alpha = 0.12 + 0.45 * max(0.0, 1.0 - dist * 5)
                    segs.append([pts[k], pts[k + 1]])
                    seg_colors.append((0.67, 0.28, 0.74, alpha))

        if segs:
            ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=1.4, zorder=9))

    # ------------------------------------------------------------------ agents

    def _draw_agents(self, ax, agent_positions: dict[str, tuple[int, int]],
                     strategic_targets: dict[str, tuple[int, int]],
                     prev_positions: list[dict[str, tuple[int, int]]] | None,
                     step_idx: int) -> None:
        """Agents with fading trails (width + alpha decay), directional arrow, target glow."""
        t = self._t
        agents = sorted(agent_positions.keys())
        for i, agent in enumerate(agents):
            color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            label = _AGENT_LABELS[i % len(_AGENT_LABELS)]
            r, c = agent_positions[agent]
            ax_, ay = self._cell_center(r, c)

            if prev_positions and len(prev_positions) >= 2:
                trail = []
                for pp in prev_positions[-10:]:
                    if agent in pp:
                        pr, pc_ = pp[agent]
                        trail.append(self._cell_center(pr, pc_))
                n = len(trail)
                if n >= 2:
                    for k in range(n - 1):
                        frac = k / max(n - 1, 1)
                        alpha = 0.06 + 0.40 * frac
                        lw = 0.8 + 1.8 * frac
                        ax.plot(
                            [trail[k][0], trail[k + 1][0]],
                            [trail[k][1], trail[k + 1][1]],
                            color=color, alpha=alpha, linewidth=lw,
                            solid_capstyle="round", zorder=9,
                        )
                    if n >= 2:
                        dx = trail[-1][0] - trail[-2][0]
                        dy = trail[-1][1] - trail[-2][1]
                        ln = math.hypot(dx, dy)
                        if ln > 0:
                            sc = self._cell_half_w * 0.5
                            ax.annotate(
                                "", xy=(trail[-1][0] + dx / ln * sc, trail[-1][1] + dy / ln * sc),
                                xytext=trail[-1],
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                                mutation_scale=8),
                                zorder=9,
                            )

            tar = strategic_targets.get(agent)
            if tar:
                tr, tc = tar
                tx, ty = self._cell_center(tr, tc)
                ax.plot([ax_, tx], [ay, ty], color="#AB47BC", linestyle="--",
                        linewidth=0.9, alpha=0.40, zorder=9)
                pulse_r = self._cell_half_w * (0.55 + 0.15 * math.sin(step_idx * 0.7))
                pulse_a = 0.22 + 0.12 * math.sin(step_idx * 0.7)
                ax.add_patch(mpatches.Circle(
                    (tx, ty), pulse_r, facecolor="#AB47BC10",
                    edgecolor=(*matplotlib.colors.to_rgb("#AB47BC"), pulse_a),
                    linewidth=1.3, zorder=9,
                ))
                ax.scatter(tx, ty, marker="X", s=75 * t["fs"],
                           c="#AB47BC", edgecolors="white", linewidths=0.7,
                           alpha=0.85, zorder=10)

            ax.add_patch(mpatches.Circle(
                (ax_, ay), self._cell_half_w * 0.50,
                facecolor="none", edgecolor=color, linewidth=0.7, alpha=0.25, zorder=10,
            ))
            ax.add_patch(mpatches.Circle(
                (ax_, ay), self._cell_half_w * 0.38,
                facecolor="white", edgecolor="none", alpha=0.70, zorder=10.5,
            ))
            ax.scatter(ax_, ay, marker="o", s=180 * t["fs"], c=color,
                       edgecolors="white", linewidths=1.8, zorder=11)
            ax.annotate(label, (ax_, ay), fontsize=7 * t["fs"], fontweight="bold",
                        color="white", ha="center", va="center", zorder=12)

    # ================================================================== decorations

    def _draw_title(self, ax, policy_name: str, fs: float) -> None:
        region_label = self.meta["display_name"]
        ax.set_title(policy_name, fontsize=12 * fs, fontweight="bold", color="white",
                     pad=10 * fs, loc="left")
        ax.text(1.0, 1.025, region_label, transform=ax.transAxes,
                fontsize=7.5 * fs, color="#aaaaaa", ha="right", va="bottom")

    def _draw_scale_bar(self, ax, fs: float) -> None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span = y1 - y0
        bar_len, n_seg = 100_000, 4
        seg_len = bar_len / n_seg
        bx = x0 + (x1 - x0) * 0.05
        by = y0 + span * 0.04
        th = span * 0.005

        for s in range(n_seg):
            c = "white" if s % 2 == 0 else "#333333"
            sx = bx + s * seg_len
            ax.fill_between([sx, sx + seg_len], by - th, by + th,
                            color=c, zorder=15, edgecolor="white", linewidth=0.3)

        for val, off in [(0, 0), (50, 0.5), (100, 1.0)]:
            lbl = f"{val}" if val < 100 else "100 km"
            ax.text(bx + bar_len * off, by - th * 2.8, lbl, color="white",
                    fontsize=4.5 * fs, ha="center", va="top", fontweight="bold", zorder=15)

    def _draw_north_arrow(self, ax, fs: float) -> None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span = y1 - y0
        nx = x1 - (x1 - x0) * 0.05
        ny = y1 - span * 0.06
        ah = span * 0.035
        aw = (x1 - x0) * 0.011

        ax.add_patch(plt.Polygon(
            [(nx + aw * 0.08, ny - ah + span * 0.002),
             (nx - aw + aw * 0.08, ny - ah * 1.7 + span * 0.002),
             (nx + aw + aw * 0.08, ny - ah * 1.7 + span * 0.002)],
            closed=True, facecolor="#00000030", edgecolor="none", zorder=14,
        ))
        ax.add_patch(plt.Polygon(
            [(nx, ny - ah), (nx - aw, ny - ah * 1.7), (nx + aw, ny - ah * 1.7)],
            closed=True, facecolor="white", edgecolor="#cccccc", linewidth=0.4, zorder=15,
        ))
        ax.text(nx, ny - ah * 0.70, "N", color="white", fontsize=7.5 * fs,
                fontweight="bold", ha="center", va="bottom", zorder=15)

    def _draw_attribution(self, ax) -> None:
        txt = AnchoredText(
            self.tile_info["attribution"], loc="lower right",
            prop=dict(size=4.5, color="#999999"), frameon=True, borderpad=0.3, pad=0.2,
        )
        txt.patch.set_facecolor("#00000080")
        txt.patch.set_edgecolor("none")
        ax.add_artist(txt)

    # ================================================================== panels

    def _draw_metrics_panel(
        self, fig, step: int, wel: float, isr: float, ce: float,
        burned: int, suppressed: int, protected: int, total_assets: int, fs: float,
    ) -> None:
        t = self._t
        p = fig.add_axes(t["panel_rect"])
        p.set_xlim(0, 1)
        p.set_ylim(0, 1)
        p.set_facecolor(t["panel_bg"])
        p.patch.set_alpha(0.92)
        for sp in p.spines.values():
            sp.set_edgecolor(t["panel_border"])
            sp.set_linewidth(t["border_lw"])
        p.set_xticks([])
        p.set_yticks([])

        sim_m = step * 30
        sh, sm = divmod(sim_m, 60)

        p.text(0.01, 0.80, f"STEP {step:03d}", fontsize=8.5 * fs, color="#E0E0E0",
               va="center", ha="left", fontweight="bold", fontfamily="monospace")
        p.text(0.20, 0.80, f"T+{sh}h{sm:02d}m", fontsize=6.5 * fs, color="#777799",
               va="center", ha="left", fontfamily="monospace")

        p.plot([0.0, 1.0], [0.55, 0.55], color=t["panel_border"], linewidth=0.4,
               transform=p.transData)

        wel_c = "#76FF03" if wel < 5 else "#FFD600" if wel < 15 else "#FF3D00"
        isr_c = "#76FF03" if isr > 0.8 else "#FFD600" if isr > 0.5 else "#FF3D00"

        metrics = [
            ("WEL", f"{wel:.1f}", 0.01, wel_c),
            ("ISR", f"{isr:.2f}", 0.16, isr_c),
            ("CE",  f"{ce:.2f}", 0.31, "#B0BEC5"),
            ("BRN", f"{burned}", 0.46, "#FF3D00"),
            ("SUP", f"{suppressed}", 0.61, "#00D9FF"),
            ("DEF", f"{protected}/{total_assets}", 0.78, "#76FF03"),
        ]
        for lbl, val, x, color in metrics:
            p.text(x, 0.35, lbl, fontsize=5 * fs, color="#888899", va="center",
                   ha="left", fontweight="bold", fontfamily="monospace")
            p.text(x, 0.10, val, fontsize=7.5 * fs, color=color, va="center",
                   ha="left", fontweight="bold", fontfamily="monospace")

    def _draw_legend(self, fig, fs: float) -> None:
        t = self._t
        lr = list(t["legend_rect"])
        lr[2] *= 0.82
        lr[3] *= 0.82
        lr[0] += (t["legend_rect"][2] - lr[2])
        lr[1] += (t["legend_rect"][3] - lr[3])

        lg = fig.add_axes(lr)
        lg.set_xlim(0, 1)
        lg.set_ylim(0, 1)
        lg.set_facecolor(t["panel_bg"])
        lg.patch.set_alpha(0.92)
        for sp in lg.spines.values():
            sp.set_edgecolor(t["panel_border"])
            sp.set_linewidth(t["border_lw"])
        lg.set_xticks([])
        lg.set_yticks([])

        col1 = [
            ("█", "#FFD600", "New fire"),
            ("█", "#B71C1C", "Old fire"),
            ("█", "#8DEDFF", "Firebreak"),
        ]
        col2 = [
            ("●", _AGENT_COLORS[0], "Agent"),
        ]

        marker = self.meta["asset_marker"]
        acolor = self.meta["asset_color"]
        alabel = self.meta["asset_label"]
        col2.append(("", acolor, alabel))
        col2.append(("×", "#AB47BC", "Target"))

        for i, (sym, color, label) in enumerate(col1):
            y = 0.82 - i * 0.30
            lg.text(0.05, y, sym, fontsize=6.5 * fs, color=color, va="center", fontfamily="monospace")
            lg.text(0.17, y, label, fontsize=5 * fs, color="#cccccc", va="center")

        for i, (sym, color, label) in enumerate(col2):
            y = 0.82 - i * 0.30
            if sym:
                lg.text(0.56, y, sym, fontsize=6.5 * fs, color=color, va="center", fontfamily="monospace")
            else:
                lg.scatter([0.58], [y], marker=marker, s=35 * fs, c="#FFFFFF",
                           edgecolors=color, linewidths=0.8, transform=lg.transData,
                           clip_on=False, zorder=5)
            lg.text(0.67, y, label, fontsize=5 * fs, color="#cccccc", va="center")
