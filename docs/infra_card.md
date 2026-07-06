# Infrastructure Card

This card documents the source coordinates, licensing, values, and sensitivity ranges of the critical infrastructure layers implemented in Phase 3 of the project.

## 1. Asset Coordinates & Types

Approximate public coordinates of critical infrastructure assets are utilized for both regions to ensure the multi-agent policies learn to coordinate risk-aware protection.

### Saudi Arabia (Eastern-Province Petroleum Sites)
Public petroleum infrastructure points in the Eastern Province (Ghawar field, oil hubs, and processing sites):
* **Refinery (Abqaiq / Buqayq):** `(49.67, 25.93)` — World's largest crude processing facility.
* **Refinery & Export Terminal (Ras Tanura):** `(50.16, 26.64)` — Major refinery and loading port.
* **Pipeline / Gathering Node (Ghawar / Uthmaniyah):** `(49.30, 25.40)` — Giant oil field hub.
* **Pipeline / Gathering Node (Khurais):** `(48.40, 25.10)` — Major oil field hub.
* **Storage Hub (Qatif):** `(50.00, 26.55)` — Oil field storage facility.
* **Storage Hub (Safaniya):** `(48.80, 28.00)` — World's largest offshore field terminal.
* **Industrial Site (Dhahran / Dammam):** `(50.10, 26.40)` — Aramco Headquarters / industrial hub.

### California (Critical Facilities & WUI Zones)
Representative high-value assets situated in Northern California to enable symmetric transfer study:
* **Refinery / High-Value Facility (Susanville Area):** `(-120.66, 40.42)`
* **Storage / Critical Facility (Redding):** `(-122.39, 40.59)`
* **Storage / Critical Facility (Santa Rosa):** `(-122.71, 38.44)`
* **Storage / Critical Facility (Eureka):** `(-124.16, 40.80)`
* **Industrial / Urban Cluster (Sacramento):** `(-121.49, 38.58)`
* **Industrial / Urban Cluster (Chico):** `(-121.84, 39.73)`

---

## 2. Asset Values & Classification Codes

Assets are classified into four codes. Higher values reflect higher economic/operational criticality:

| Code | Asset Type | Default Value | Sensitivity Range | Default Blast Radius (cells) |
|------|------------|---------------|-------------------|-----------------------------|
| 1 | Refinery | 10.0 | `[5.0 - 20.0]` | 3 |
| 2 | Pipeline | 4.0 | `[2.0 - 8.0]` | 2 |
| 3 | Storage | 6.0 | `[3.0 - 12.0]` | 3 |
| 4 | Industrial | 3.0 | `[1.0 - 6.0]` | 2 |

---

## 3. Criticality Heatmap (Gaussian Falloff)

Rather than treating assets as single pixels, a continuous criticality field is constructed via Gaussian decay to model surrounding vulnerability:
$$C(x, y) = \sum_{a} V_a \exp\left(-\frac{(x - x_a)^2 + (y - y_a)^2}{2\sigma^2}\right)$$
where $V_a$ is the asset value, and $\sigma$ represents the falloff radius (default $\sigma = 1.0$). The final raster is normalized to `[0, 1]` at its peak.

---

## 4. Post-Spread Cascade Detonation

If fire reaches a high-value asset cell (`fire_intensity > 0.1`), there is a risk of catastrophic cascading detonation:
* **Trigger:** Asset cell fire intensity exceeds $0.1$.
* **Probability:** $p_{cascade}$ (default $0.0$, configurable up to $1.0$ for sensitivity analyses).
* **Propagation:** Nearby burnable cells within the asset's Chebyshev/circular `blast_radius` are ignited at full intensity ($1.0$).
* **Metric:** Tracked via **Critical Cascade Loss (CCL)**, representing the total count of cells ignited by detonations.

---

## 5. Licensing & Provenance

* **Saudi Eastern-Province sites:** Coords sourced from OpenStreetMap (OSM) public database queries for industrial facilities, under the Open Database License (ODbL).
* **California sites:** Coords sourced from USGS Geographic Names Information System (GNIS) and public state critical infrastructure registers (Public Domain).
