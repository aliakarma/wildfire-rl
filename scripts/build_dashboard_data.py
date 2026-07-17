"""Build the dashboard data layer from the frozen result artifacts.

The only writer of ``dashboard/public/data/``. Reads the frozen artifacts
(``wildfire_phase3_multiseed/``, ``wildfire_phase4/``, ``wildfire_phase6/``), verifies the
freeze fingerprint first (same computation as ``scripts/freeze_results.py``; aborts on
mismatch), and emits schema-checked JSON the static site consumes. Numbers are emitted at
full precision — rounding happens only at render time — and labels are emitted as keys
(``hiercomm_heur``, ``mappo``, …); display names come from the dashboard i18n layer.

    python scripts/build_dashboard_data.py                # emit dashboard/public/data/*.json
    python scripts/build_dashboard_data.py --copy-media   # also copy rollout GIFs/snapshots

Stdlib only, platform-agnostic (runs under the WSL venv or Windows Python alike).
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASE3 = REPO / "wildfire_phase3_multiseed"
PHASE4 = REPO / "wildfire_phase4"
PHASE6 = REPO / "wildfire_phase6"
GIFS5 = REPO / "figures" / "wildfire_phase5_gifs"
GIFS6 = REPO / "figures" / "wildfire_phase6_gifs"
PAPER_FIGS = REPO / "AAAI Template" / "Figures"
REGIMES_PY = REPO / "src" / "wildfire_marl" / "env" / "regimes.py"
OUT_DATA = REPO / "dashboard" / "public" / "data"
OUT_MEDIA = REPO / "dashboard" / "public" / "media"

#: Every artifact directory the dashboard reads is fingerprint-gated. A re-freeze of any
#: directory must update its entry here (and nowhere else — the consistency checker
#: imports this dict).
EXPECTED_FINGERPRINTS: dict[str, str] = {
    "wildfire_phase3_multiseed": (
        "da4a381dc6ab43e5cb2a01b60fd5c8f32f2b111587fa38b0277b100fc35586dd"
    ),
    "wildfire_phase4": "be6d0e218ffe77f144a9171ac21ec5562c422cf9180c1d543fa150a11c36f94d",
    "wildfire_phase6": "cfce760842dd1e2f67d1a0f9709b8be4c701c464353d69d81d884626ba9af93b",
}

# Fixed policy order (paper Table 1); every consumer preserves it.
POLICY_ORDER = [
    "noop",
    "value_first",
    "greedy_risk",
    "local_reactive",
    "mappo",
    "commnet",
    "hiercomm_heur",
]
LEARNED = {"mappo", "commnet", "hiercomm_heur"}
REGIONS = ["saudi", "california"]
TRAIN_SEEDS = [42, 1042, 2042, 3042, 4042]
CURVE_MAX_POINTS = 400
CURVE_ROLL_WINDOW = 25  # 25-episode rolling mean, exactly like the paper figure

_SKIP = {"MANIFEST.sha256", "FREEZE.json"}


# ---------------------------------------------------------------------------
# Fingerprint gate (mirrors scripts/freeze_results.py)
# ---------------------------------------------------------------------------


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_fingerprint(results_dir: Path) -> str:
    files = sorted(p for p in results_dir.rglob("*") if p.is_file() and p.name not in _SKIP)
    manifest = "".join(f"{_sha256(p)}  {p.relative_to(results_dir).as_posix()}\n" for p in files)
    return hashlib.sha256(manifest.encode()).hexdigest()


def verify_freeze() -> dict:
    """Verify every gated artifact directory; return the phase-3 freeze record."""
    freezes: dict[str, dict] = {}
    for dirname, expected in EXPECTED_FINGERPRINTS.items():
        root = REPO / dirname
        freeze = json.loads((root / "FREEZE.json").read_text())
        recorded = freeze["fingerprint_sha256"]
        if recorded != expected:
            raise SystemExit(
                f"{dirname}/FREEZE.json fingerprint {recorded[:16]}… != expected "
                f"{expected[:16]}…"
            )
        actual = compute_fingerprint(root)
        if actual != expected:
            raise SystemExit(
                f"{dirname}: recomputed fingerprint {actual[:16]}… does not match the frozen "
                f"record {expected[:16]}… — artifacts were modified; refusing to build."
            )
        print(f"  OK {dirname} {expected[:16]}… ({freeze['n_files']} files)")
        freezes[dirname] = freeze
    return freezes["wildfire_phase3_multiseed"]


def _git(*args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=REPO, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Emitters (one per output file)
# ---------------------------------------------------------------------------


def build_meta(freeze: dict) -> dict:
    return {
        "fingerprint": freeze["fingerprint_sha256"],
        "fingerprints": dict(EXPECTED_FINGERPRINTS),
        "frozen_utc": freeze["frozen_utc"],
        "freeze_commit": freeze["git_sha"],
        "commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "protocol": {
            "train_seeds": TRAIN_SEEDS,
            "episodes": 15,
            "unit": "per-training-seed mean (n=5)",
        },
    }


def build_main_results(meta: dict) -> dict:
    src = json.loads((PHASE3 / "phase3_summary.json").read_text())
    regions: dict = {}
    for region in REGIONS:
        block = src["regions"][region]
        policies = {}
        for key in POLICY_ORDER:
            metrics = {}
            for name, m in block[key]["metrics"].items():
                metrics[name] = {
                    "mean": m["mean"],
                    "std": m["std"],
                    "ci": [m["ci_lo"], m["ci_hi"]],
                    "n_train_seeds": m["n_train_seeds"],
                    "values": m["values"],
                }
            policies[key] = {"learned": key in LEARNED, **metrics}
        regions[region] = {"policies": policies, "comparisons": block["comparisons"]}
    return {"meta": meta, "policy_order": POLICY_ORDER, "regions": regions}


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    out: list[float] = []
    acc = 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= window:
            acc -= xs[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _downsample(idx: list[int], n: int) -> list[int]:
    if len(idx) <= n:
        return idx
    step = (len(idx) - 1) / (n - 1)
    return [idx[round(i * step)] for i in range(n)]


def build_train_curves(meta: dict) -> dict:
    """Per-method training curves, aligned by episode.

    Seeds finish different episode counts within the 100k-step budget (episodes can end
    early), so cross-seed aggregation is done **by episode index** and the x-axis is the
    episode number — labeling it with any single seed's ``env_steps`` would misstate the
    alignment. ``env_steps_mean`` (mean cumulative steps across seeds at that episode) is
    kept for tooltip context, and ``n_episodes``/``episodes_per_seed`` document how much
    of each seed's run the common range covers.
    """
    curves: dict = {}
    for region in REGIONS:
        curves[region] = {}
        for method in sorted(LEARNED):
            per_seed: dict[str, list[float]] = {}
            steps_per_seed: dict[str, list[int]] = {}
            for seed in TRAIN_SEEDS:
                path = PHASE3 / f"train_curve_checkpoint_{method}_{region}_s{seed}.csv"
                with open(path, newline="") as f:
                    rows = list(csv.DictReader(f))
                per_seed[str(seed)] = _rolling_mean(
                    [float(r["WEL"]) for r in rows], CURVE_ROLL_WINDOW
                )
                steps_per_seed[str(seed)] = [int(r["env_steps"]) for r in rows]
            n = min(len(v) for v in per_seed.values())
            keep = _downsample(list(range(n)), CURVE_MAX_POINTS)
            median = [statistics.median(per_seed[str(s)][i] for s in TRAIN_SEEDS) for i in keep]
            curves[region][method] = {
                "episode": [i + 1 for i in keep],
                "env_steps_mean": [
                    round(statistics.mean(steps_per_seed[str(s)][i] for s in TRAIN_SEEDS))
                    for i in keep
                ],
                "median": median,
                "seeds": {s: [per_seed[s][i] for i in keep] for s in per_seed},
                "rolling_window": CURVE_ROLL_WINDOW,
                "n_episodes": n,
                "episodes_per_seed": {s: len(v) for s, v in per_seed.items()},
            }
    return {"meta": meta, "regions": curves}


def build_passthrough(meta: dict, path: Path) -> dict:
    """Mirror a frozen summary verbatim under a shared meta block."""
    src = json.loads(path.read_text())
    return {"meta": meta, "source": path.relative_to(REPO).as_posix(), **src}


def _module_literal(tree: ast.Module, name: str) -> dict:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise SystemExit(f"{REGIMES_PY}: could not find literal assignment for {name!r}")


def build_benchmark(meta: dict) -> dict:
    """Regime parameter table, extracted from the env source so the docs cannot drift.

    ``regimes.py`` is parsed as an AST literal (no import — the env stack must not be a
    build dependency). Saudi uses the base ``REGIMES``; California applies its overrides.
    """
    tree = ast.parse(REGIMES_PY.read_text())
    regimes = _module_literal(tree, "REGIMES")
    overrides = _module_literal(tree, "REGION_OVERRIDES")
    sweep = ["easy", "medium", "hard"]
    merged = {
        "saudi": {r: regimes[r] for r in sweep},
        "california": {
            r: {**regimes[r], **overrides.get("california", {}).get(r, {})} for r in sweep
        },
    }
    return {
        "meta": meta,
        "source": REGIMES_PY.relative_to(REPO).as_posix(),
        "regimes": merged,
    }


def build_media_index(meta: dict) -> dict:
    """Inventory the pre-rendered rollout media that actually exists on disk.

    phase5 has no CommNet/Value-First/Greedy-Risk rollouts; CommNet appears via its phase6
    native rollouts instead (per the guide's known-gap rule: never ship a picker with a
    silently missing option).
    """
    items: list[dict] = []

    def add(kind: str, region: str, policy: str, src: Path, poster: Path | None, **extra):
        if not src.exists():
            return
        # Prefer the MP4 sibling (scripts/compress_media.py) — 10–20× smaller than the
        # GIF; the GIF stays available as a download fallback.
        mp4 = src.with_suffix(".mp4")
        best = mp4 if mp4.exists() else src
        entry = {
            "id": src.stem,
            "kind": kind,
            "region": region,
            "policy": policy,
            "src": f"media/{best.parent.name}/{best.name}",
            "bytes": best.stat().st_size,
            **extra,
        }
        if best is not src:
            entry["gif_fallback"] = f"media/{src.parent.name}/{src.name}"
        if poster is not None and poster.exists():
            entry["poster"] = f"media/{poster.parent.name}/{poster.name}"
        items.append(entry)

    for region in REGIONS:
        add(
            "comparison",
            region,
            "all",
            GIFS5 / f"comparison_grid_{region}.gif",
            GIFS5 / f"comparison_snapshot_{region}.png",
        )
        for policy in ["noop", "local_reactive", "mappo", "commnet", "hiercomm_heur"]:
            add(
                "rollout",
                region,
                policy,
                GIFS5 / f"rollout_{policy}_{region}.gif",
                GIFS5 / f"snapshot_{policy}_{region}.png",
            )
        for policy in ["commnet", "hiercomm_heur"]:
            other = "california" if region == "saudi" else "saudi"
            add(
                "transfer_native",
                region,
                policy,
                GIFS6 / f"rollout_{policy}_{region}_native.gif",
                GIFS6 / f"fig_transfer_{policy}_{region}.png",
                direction=f"{region}->{region}",
            )
            add(
                "transfer_cross",
                region,
                policy,
                GIFS6 / f"rollout_{policy}_{region}_to_{other}.gif",
                GIFS6 / f"fig_transfer_{policy}_{region}.png",
                direction=f"{region}->{other}",
            )
    return {"meta": meta, "items": items}


def build_repro(meta: dict) -> dict:
    """Reproducibility payload: seed-42 checkpoint hashes (paper Appendix D) + downloads."""
    hashes: dict[str, str] = {}
    for line in (PHASE3 / "MANIFEST.sha256").read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name.startswith("checkpoint_") and name.endswith("_s42.pt"):
            hashes[name] = digest
    return {
        "meta": meta,
        "seed42_checkpoint_sha256": hashes,
        "n_checkpoints": sum(
            1 for line in (PHASE3 / "MANIFEST.sha256").read_text().splitlines() if ".pt" in line
        ),
        "reproduction_command": (
            "python scripts/run_phase3.py \\\n"
            "  --methods noop,value_first,greedy_risk,local_reactive,mappo,commnet,hiercomm_heur \\\n"
            "  --regions saudi,california --train-seeds 42,1042,2042,3042,4042 --parallel 6 \\\n"
            "  --train-steps 100000 --episodes 15 --out wildfire_phase3_multiseed"
        ),
        "verify_command": "python scripts/freeze_results.py wildfire_phase3_multiseed",
        "compute": "CPU-only (Ryzen 5 5600H), WSL2 Ubuntu; bit-identical CPU determinism",
    }


def copy_downloads() -> int:
    """Copy the small frozen CSV/manifest artifacts for one-click download."""
    dst_dir = OUT_DATA / "downloads"
    dst_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        PHASE3 / "phase3_raw.csv",
        PHASE3 / "phase3_summary.json",
        PHASE3 / "MANIFEST.sha256",
        PHASE3 / "FREEZE.json",
        PHASE4 / "ablation_results.csv",
        PHASE4 / "ablation_summary.json",
        PHASE4 / "robustness_results.csv",
        PHASE4 / "robustness_summary.json",
        PHASE6 / "generalization_raw.csv",
        PHASE6 / "generalization_summary.json",
        PHASE6 / "transfer_matrix_raw.csv",
        PHASE6 / "transfer_summary.json",
    ]
    for src in sources:
        shutil.copy2(src, dst_dir / src.name)
    return len(sources)


def copy_media() -> int:
    """Copy rollout media + the two paper figures into dashboard/public/media/.

    The media folder is gitignored (the GIFs total ~460 MB — repo policy forbids committing
    large binaries); this copy step is re-run before any site build/deploy.
    """
    n = 0
    for src_dir in (GIFS5, GIFS6):
        dst_dir = OUT_MEDIA / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_dir.iterdir()):
            if p.suffix.lower() in {".gif", ".png", ".mp4"}:
                dst = dst_dir / p.name
                if not dst.exists() or dst.stat().st_size != p.stat().st_size:
                    shutil.copy2(p, dst)
                n += 1
    paper_dst = OUT_MEDIA / "paper"
    paper_dst.mkdir(parents=True, exist_ok=True)
    for name in ("qualitative_comparison_california.png", "learned_training_curves.png"):
        src = PAPER_FIGS / name
        if src.exists():
            shutil.copy2(src, paper_dst / name)
            n += 1
    pdf = PAPER_FIGS.parent / "AnonymousSubmission2027.pdf"
    if pdf.exists():
        shutil.copy2(pdf, paper_dst / "paper.pdf")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Lightweight schema checks (structure only; value checks live in
# scripts/check_dashboard_consistency.py)
# ---------------------------------------------------------------------------


def check_schemas(outputs: dict[str, dict]) -> None:
    mr = outputs["main_results.json"]
    for region in REGIONS:
        pols = mr["regions"][region]["policies"]
        assert list(pols) == POLICY_ORDER, f"policy order broken in {region}"
        for key, pol in pols.items():
            for metric in ("WEL", "ISR", "CE"):
                m = pol[metric]
                assert set(m) >= {"mean", "std", "ci", "values"}, f"{region}/{key}/{metric}"
                assert len(m["ci"]) == 2
        comps = mr["regions"][region]["comparisons"]["WEL"]
        assert all(set(c) >= {"t", "p", "d", "test"} for c in comps.values())
    tc = outputs["train_curves.json"]
    for region in REGIONS:
        for method in LEARNED:
            c = tc["regions"][region][method]
            assert (len(c["episode"]) == len(c["median"]) == len(c["env_steps_mean"])) and len(
                c["median"]
            ) <= CURVE_MAX_POINTS
            assert set(c["seeds"]) == {str(s) for s in TRAIN_SEEDS}
            assert c["n_episodes"] == min(c["episodes_per_seed"].values())
    for name in ("ablations.json", "robustness.json", "generalization.json", "transfer.json"):
        assert "meta" in outputs[name] and "protocol" in outputs[name], name
    assert all("src" in i for i in outputs["media.json"]["items"])
    bench = outputs["benchmark.json"]["regimes"]
    for region in REGIONS:
        for regime in ("easy", "medium", "hard"):
            assert {"wind_scale", "ffmc", "treat_radius"} <= set(bench[region][regime])


def _strip_volatile(payload: dict) -> dict:
    """Copy of a payload with the build date blanked, for change detection."""
    clone = json.loads(json.dumps(payload))
    if "built" in clone:
        clone["built"] = ""
    if isinstance(clone.get("meta"), dict):
        clone["meta"]["built"] = ""
    return clone


def write_output(path: Path, payload: dict) -> bool:
    """Write JSON, but keep the existing file (and its build date) when nothing else
    changed — so committed data does not churn on every regeneration."""
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            old = None
        if old is not None and _strip_volatile(old) == _strip_volatile(payload):
            return False
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--copy-media",
        action="store_true",
        help="also copy rollout GIFs/snapshots + paper figures into dashboard/public/media/",
    )
    args = ap.parse_args()

    print("Verifying freeze fingerprints …")
    freeze = verify_freeze()

    meta = build_meta(freeze)
    outputs: dict[str, dict] = {
        "meta.json": meta,
        "main_results.json": build_main_results(meta),
        "train_curves.json": build_train_curves(meta),
        "ablations.json": build_passthrough(meta, PHASE4 / "ablation_summary.json"),
        "robustness.json": build_passthrough(meta, PHASE4 / "robustness_summary.json"),
        "generalization.json": build_passthrough(meta, PHASE6 / "generalization_summary.json"),
        "transfer.json": build_passthrough(meta, PHASE6 / "transfer_summary.json"),
        "benchmark.json": build_benchmark(meta),
        "media.json": build_media_index(meta),
        "reproducibility.json": build_repro(meta),
    }
    check_schemas(outputs)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = OUT_DATA / name
        changed = write_output(path, payload)
        status = "wrote" if changed else "unchanged"
        print(f"  {status} {path.relative_to(REPO)} ({path.stat().st_size:,} B)")

    n_dl = copy_downloads()
    print(f"  copied {n_dl} download artifacts -> {OUT_DATA.relative_to(REPO)}/downloads")

    if args.copy_media:
        n = copy_media()
        print(f"  copied {n} media files -> {OUT_MEDIA.relative_to(REPO)}")

    print("Dashboard data build complete.")


if __name__ == "__main__":
    main()
