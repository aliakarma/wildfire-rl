# Results dashboard

An interactive presentation of every result in the paper, reading the same frozen JSON that
`scripts/build_dashboard_data.py` generates from the `wildfire_phase*/` directories.

## Run it (no installation)

```bash
python -m http.server 8000 --directory dashboard/dist
# open http://localhost:8000
```

**Must be served over HTTP.** Opening `dist/index.html` as a `file://` URL yields a blank page:
the app fetches its data at runtime and browsers block those requests for local files.

## Run from source

```bash
cd dashboard
npm ci
cp -r dist/data public/          # data ships once, inside dist/
cp -r dist/media public/         # optional: videos and posters
npm run dev
```

Only `dist/` carries the data and media in this archive, to avoid shipping both copies under
the 50 MB cap. The two `cp` commands restore the layout Vite expects for a dev build.

## Pages

| Page | What it shows |
|---|---|
| Results | Main table — WEL and ISR for all 7 policies across both regions, with across-seed error bars |
| Benchmark | Per-policy comparison, including the information-matched heuristics |
| Ablations | Single-factor ablations; which components actually drive performance |
| Robustness | Extended difficulty sweep (easy / medium / hard) for all seven policies |
| Generalization | Cross-regional generalization sweep |
| Transfer | Saudi ↔ California transfer matrix, TRS, and adaptation asymmetry |
| Replays | Step-by-step episode replays with agent positions and fire state |
| Reproducibility | Freeze fingerprints, protocol, and seed configuration |

The interface is bilingual (English / Arabic, with RTL layout) and supports light and dark
themes.

## Data provenance

Every JSON under `dist/data/` is generated from the frozen results and carries the
`fingerprint_sha256` of the results directory it came from, in its `meta` block. To confirm the
dashboard shows the same numbers as the paper:

```bash
python scripts/check_dashboard_consistency.py
```

## Media

Videos were re-encoded (H.264, width capped at 960 px) and posters converted to JPEG to fit the
supplement's size cap; `media.json` records the resulting byte counts. The source GIFs are
excluded and regenerable with `scripts/render_phase5_gifs.py` and
`scripts/render_phase6_transfer.py`.
