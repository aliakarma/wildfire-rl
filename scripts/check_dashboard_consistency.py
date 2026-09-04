"""Assert dashboard JSON == frozen summaries == the paper's headline values.

Release gate for the dashboard (Dashboard_Guide.md §4.3): every value in
``dashboard/public/data/*.json`` must match the frozen artifacts exactly, and the paper's
headline numbers must be present verbatim. A mismatch exits non-zero and fails the build —
making a paper/dashboard contradiction structurally impossible.

    python scripts/check_dashboard_consistency.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "dashboard" / "public" / "data"
PUBLIC = REPO / "dashboard" / "public"
PHASE3 = REPO / "wildfire_phase3_multiseed"
PHASE4 = REPO / "wildfire_phase4"
PHASE6 = REPO / "wildfire_phase6"

# Single source of truth for the expected fingerprints lives in the build script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dashboard_data import EXPECTED_FINGERPRINTS  # noqa: E402

EXPECTED_FINGERPRINT = EXPECTED_FINGERPRINTS["wildfire_phase3_multiseed"]

_failures: list[str] = []


def _fail(msg: str) -> None:
    _failures.append(msg)
    print(f"  FAIL {msg}")


def _ok(msg: str) -> None:
    print(f"  ok   {msg}")


def _check(label: str, got: float, want: float, tol: float = 1e-12) -> None:
    if math.isclose(got, want, rel_tol=0, abs_tol=tol):
        _ok(f"{label} = {got}")
    else:
        _fail(f"{label}: dashboard {got} != frozen {want}")


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def check_exact_mirrors() -> None:
    """Dashboard JSON must carry the frozen summaries' values exactly."""
    src = json.loads((PHASE3 / "phase3_summary.json").read_text())
    dash = _load("main_results.json")
    for region, block in src["regions"].items():
        for policy, entry in block.items():
            if policy == "comparisons":
                if dash["regions"][region]["comparisons"] != entry:
                    _fail(f"main_results comparisons diverge in {region}")
                continue
            for metric, m in entry["metrics"].items():
                d = dash["regions"][region]["policies"][policy][metric]
                if (
                    d["mean"] != m["mean"]
                    or d["std"] != m["std"]
                    or d["ci"] != [m["ci_lo"], m["ci_hi"]]
                    or d["values"] != m["values"]
                ):
                    _fail(f"main_results {region}/{policy}/{metric} diverges")
    _ok("main_results.json mirrors phase3_summary.json")

    for name, path in [
        ("ablations.json", PHASE4 / "ablation_summary.json"),
        ("robustness.json", REPO / "wildfire_phase4_extended" / "robustness_summary.json"),
        ("generalization.json", PHASE6 / "generalization_summary.json"),
        ("transfer.json", PHASE6 / "transfer_summary.json"),
    ]:
        frozen = json.loads(path.read_text())
        emitted = _load(name)
        stripped = {k: v for k, v in emitted.items() if k not in ("meta", "source")}
        if stripped == frozen:
            _ok(f"{name} mirrors {path.name}")
        else:
            _fail(f"{name} diverges from {path.name}")


def check_headlines() -> None:
    """Spot-assert the paper's headline numbers (docs/RESULTS_FROZEN.md, Table 1)."""
    mr = _load("main_results.json")
    saudi = mr["regions"]["saudi"]["policies"]
    cal = mr["regions"]["california"]["policies"]
    _check("Saudi HierComm WEL mean", saudi["hiercomm_heur"]["WEL"]["mean"], 7.024, 1e-3)
    _check("Saudi HierComm WEL std", saudi["hiercomm_heur"]["WEL"]["std"], 2.3607, 1e-4)
    _check("California HierComm WEL mean", cal["hiercomm_heur"]["WEL"]["mean"], 5.8773, 1e-4)
    _check("California HierComm WEL std", cal["hiercomm_heur"]["WEL"]["std"], 1.0402, 1e-4)
    _check("Saudi No-Op WEL", saudi["noop"]["WEL"]["mean"], 27.0)
    _check("California No-Op WEL", cal["noop"]["WEL"]["mean"], 34.0)
    _check("Saudi HierComm ISR", saudi["hiercomm_heur"]["ISR"]["mean"], 0.8088, 1e-4)
    _check("California HierComm ISR", cal["hiercomm_heur"]["ISR"]["mean"], 0.8316, 1e-3)

    p_mappo = mr["regions"]["saudi"]["comparisons"]["WEL"]["proposed_vs_mappo"]["p"]
    if 0.31 < p_mappo < 0.33:
        _ok(f"Saudi HierComm-vs-MAPPO n.s. (p = {p_mappo:.4f})")
    else:
        _fail(f"Saudi HierComm-vs-MAPPO p = {p_mappo} not the paper's 0.32 (n.s.)")

    ab = _load("ablations.json")["regions"]
    _check("Ablation w/o comms Saudi p", ab["saudi"]["wo_comms"]["vs_full_WEL"]["p"], 0.2954, 1e-4)
    tr = _load("transfer.json")["policies"]
    _check(
        "HierComm TRS_ISR saudi->california",
        tr["hiercomm_heur"]["directions"]["saudi->california"]["TRS_ISR"],
        0.4737,
        1e-4,
    )
    _check(
        "CommNet TRS_ISR saudi->california",
        tr["commnet"]["directions"]["saudi->california"]["TRS_ISR"],
        0.0958,
        1e-4,
    )


def check_meta() -> None:
    meta = _load("meta.json")
    if meta["fingerprint"] == EXPECTED_FINGERPRINT:
        _ok(f"fingerprint {meta['fingerprint'][:16]}…")
    else:
        _fail("meta.json fingerprint != expected freeze fingerprint")
    # Every gated directory: FREEZE.json on disk and meta.fingerprints must both match
    # the expected constants (the full re-hash runs in build_dashboard_data.py).
    for dirname, expected in EXPECTED_FINGERPRINTS.items():
        recorded = json.loads((REPO / dirname / "FREEZE.json").read_text())["fingerprint_sha256"]
        if recorded != expected:
            _fail(f"{dirname}/FREEZE.json fingerprint != expected constant")
        if meta.get("fingerprints", {}).get(dirname) != expected:
            _fail(f"meta.json fingerprints[{dirname}] != expected constant")
    _ok(f"all {len(EXPECTED_FINGERPRINTS)} artifact directories fingerprint-pinned")
    for name in (
        "main_results.json",
        "train_curves.json",
        "ablations.json",
        "robustness.json",
        "generalization.json",
        "transfer.json",
        "benchmark.json",
        "media.json",
    ):
        if _load(name)["meta"]["fingerprint"] != EXPECTED_FINGERPRINT:
            _fail(f"{name} meta fingerprint mismatch")
    _ok("all outputs share the freeze fingerprint")


def check_media(release: bool) -> None:
    """Every media.json reference must exist on disk (mandatory for a release build).

    The media folder is gitignored (~460 MB of GIFs), so a fresh clone legitimately
    lacks it during development — but deploying without running
    ``build_dashboard_data.py --copy-media`` would ship a gallery of broken images.
    """
    media = _load("media.json")
    refs = [i["src"] for i in media["items"]] + [
        i["poster"] for i in media["items"] if "poster" in i
    ]
    missing = [r for r in refs if not (PUBLIC / r).exists()]
    if not missing:
        _ok(f"all {len(refs)} media references exist on disk")
    elif release:
        _fail(
            f"{len(missing)}/{len(refs)} media references missing (run "
            f"build_dashboard_data.py --copy-media before deploying); first: {missing[0]}"
        )
    else:
        print(
            f"  warn {len(missing)}/{len(refs)} media references missing on disk — OK for "
            f"development, a release build must run --copy-media (use --release to enforce)"
        )


def check_replays() -> None:
    """Re-assert every exported replay's endpoint against the frozen per-episode CSVs.

    The exporter (scripts/export_replay_frames.py) verified WEL/ISR/CE/burned at
    generation time; this re-check keeps the committed replay JSON honest in CI without
    re-running the simulator.
    """
    import csv

    index_path = DATA / "replays" / "index.json"
    if not index_path.exists():
        print("  warn no replays exported yet (data/replays/index.json missing)")
        return
    index = json.loads(index_path.read_text())

    def rows(path: Path) -> list[dict]:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    p3 = rows(PHASE3 / "phase3_raw.csv")
    p6 = rows(PHASE6 / "transfer_matrix_raw.csv")
    bad = 0
    for e in index["replays"]:
        meta = json.loads((DATA / "replays" / e["file"]).read_text())["meta"]
        f = meta["final"]
        ep, group = meta["episode"], meta["episode_seed"] - 100_000 - meta["episode"]
        if e["kind"] == "native":
            match = [
                r
                for r in p3
                if r["region"] == e["region"]
                and r["policy"] == e["policy"]
                and int(r["seed"]) == group
                and int(r["episode"]) == ep
                and (not r["train_seed"] or float(r["train_seed"]) == group)
            ]
        else:
            match = [
                r
                for r in p6
                if r["policy"] == e["policy"]
                and r["ckpt_region"] == e["ckpt_region"]
                and r["eval_region"] == e["region"]
                and int(r["eval_seed"]) == group
                and int(r["episode"]) == ep
            ]
        if not match or not (
            math.isclose(f["WEL"], float(match[0]["WEL"]), abs_tol=1e-9)
            and math.isclose(f["ISR"], float(match[0]["ISR"]), abs_tol=1e-9)
            and math.isclose(f["CE"], float(match[0]["CE"]), abs_tol=1e-9)
            and int(f["burned"]) == int(float(match[0]["burned"]))
        ):
            _fail(f"replay {e['file']} endpoint does not match the frozen record")
            bad += 1
    if not bad:
        _ok(f"all {len(index['replays'])} replays match the frozen per-episode records")
    if index.get("skipped_unverified"):
        skipped = {f"{s['policy']}/{s['region']}" for s in index["skipped_unverified"]}
        print(f"  warn {len(skipped)} rollout(s) excluded as unverified: {sorted(skipped)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--release",
        action="store_true",
        help="treat missing media files as failures (mandatory before deploying)",
    )
    args = ap.parse_args()
    if not DATA.is_dir():
        raise SystemExit(f"No dashboard data at {DATA} — run scripts/build_dashboard_data.py")
    print("Checking dashboard data consistency …")
    check_meta()
    check_exact_mirrors()
    check_headlines()
    check_media(release=args.release)
    check_replays()
    if _failures:
        print(f"\n{len(_failures)} consistency failure(s) — dashboard build must not ship.")
        sys.exit(1)
    print("\nAll consistency checks passed.")


if __name__ == "__main__":
    main()
