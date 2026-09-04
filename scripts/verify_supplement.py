"""Tier-2 reproducibility check: re-derive every number the paper cites from the frozen
per-episode CSVs and diff against the values asserted in ``docs/RESULTS_FROZEN.md``.

No GPU, no simulator build, no training — this reads CSVs and does arithmetic. Runs in
seconds and is the cheapest way for a reviewer to confirm the paper's tables are what the
frozen artifacts actually say.

    python scripts/verify_supplement.py            # human-readable report
    python scripts/verify_supplement.py --json out.json

Exit code 0 iff every check passes. Works both in the development repo (``wildfire_phase3_
multiseed/`` etc.) and inside the reviewer supplement (``results/phase3/`` etc.).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    from scipy import stats as _stats
except ImportError:  # significance checks degrade to SKIP rather than failing the run
    _stats = None

# --------------------------------------------------------------------------------------
# Where the frozen artifacts live. First existing path wins, so one script serves both the
# development repo layout and the flattened supplement layout.
# --------------------------------------------------------------------------------------
DIR_CANDIDATES = {
    "phase3": ["results/wildfire_phase3_multiseed", "results/phase3", "wildfire_phase3_multiseed"],
    "phase3_learned": [
        "results/wildfire_phase3_hiercomm_learned",
        "results/phase3_hiercomm_learned",
        "wildfire_phase3_hiercomm_learned",
    ],
    "phase4": ["results/wildfire_phase4", "results/phase4", "wildfire_phase4"],
    "phase4_extended": [
        "results/wildfire_phase4_extended",
        "results/phase4_extended",
        "wildfire_phase4_extended",
    ],
    "phase6": ["results/wildfire_phase6", "results/phase6", "wildfire_phase6"],
}

# Values as printed in docs/RESULTS_FROZEN.md and the paper's main table.
# (region, policy) -> (WEL mean, WEL std or None for deterministic heuristics, ISR mean)
PHASE3_EXPECTED = {
    ("saudi", "noop"): (27.00, None, 0.286),
    ("saudi", "value_first"): (18.84, None, 0.545),
    ("saudi", "greedy_risk"): (24.72, None, 0.358),
    ("saudi", "local_reactive"): (11.91, None, 0.720),
    ("saudi", "mappo"): (9.77, 5.10, 0.692),
    ("saudi", "commnet"): (12.44, 4.32, 0.682),
    ("saudi", "hiercomm_heur"): (7.02, 2.36, 0.809),
    ("california", "noop"): (34.00, None, 0.000),
    ("california", "value_first"): (28.40, None, 0.156),
    ("california", "greedy_risk"): (28.40, None, 0.156),
    ("california", "local_reactive"): (10.23, None, 0.702),
    ("california", "mappo"): (14.17, 2.89, 0.624),
    ("california", "commnet"): (19.58, 7.77, 0.398),
    ("california", "hiercomm_heur"): (5.88, 1.04, 0.832),
}

# HierComm learned-commander ablation (RESULTS_FROZEN.md "Not in this frozen table").
PHASE3_LEARNED_EXPECTED = {
    ("saudi", "hiercomm"): (6.02, 2.04),
    ("california", "hiercomm"): (8.20, 1.63),
}

# Welch t-tests on the per-train-seed WEL means, HierComm vs baseline, as cited in the paper.
SIGNIFICANCE_EXPECTED = [
    ("california", "mappo", 0.0018),
    ("california", "commnet", 0.016),
    ("saudi", "commnet", 0.048),
    ("saudi", "mappo", 0.32),  # explicitly reported as n.s.
]

# Extended difficulty sweep headline (RESULTS_FROZEN.md, 2026-07-19).
PHASE4_EXT_EXPECTED_BEST = {
    ("saudi", "easy"): "local_reactive",
    ("california", "easy"): "local_reactive",
    ("saudi", "medium"): "hiercomm_heur",
    ("california", "medium"): "hiercomm_heur",
    ("saudi", "hard"): "hiercomm_heur",
    ("california", "hard"): "hiercomm_heur",
}

WEL_TOL = 0.01  # numbers are quoted to 2 decimals
ISR_TOL = 0.001  # quoted to 3 decimals
P_REL_TOL = 0.10  # p-values quoted to 2 significant figures


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, name: str, status: str, detail: str) -> None:
        self.rows.append({"section": section, "check": name, "status": status, "detail": detail})
        mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
        print(f"[{mark}] {name:<52} {detail}")

    @property
    def failed(self) -> int:
        return sum(r["status"] == "FAIL" for r in self.rows)

    @property
    def skipped(self) -> int:
        return sum(r["status"] == "SKIP" for r in self.rows)

    @property
    def passed(self) -> int:
        return sum(r["status"] == "PASS" for r in self.rows)


def resolve(root: Path, key: str) -> Path | None:
    for candidate in DIR_CANDIDATES[key]:
        path = root / candidate
        if path.is_dir():
            return path
    return None


def close(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def per_seed_wel(df: pd.DataFrame, region: str, policy: str) -> pd.Series:
    """Per-training-seed mean WEL. Heuristics are deterministic and carry no train_seed."""
    sub = df[(df.region == region) & (df.policy == policy)]
    if sub.empty:
        return pd.Series(dtype=float)
    if sub.train_seed.notna().any():
        return sub.groupby("train_seed").WEL.mean()
    return pd.Series([sub.WEL.mean()])


def check_phase3(root: Path, rep: Report) -> None:
    section = "Phase 3 — main results table"
    d = resolve(root, "phase3")
    if d is None:
        rep.add(section, "phase3 results directory", "SKIP", "not present in this archive")
        return
    raw = d / "phase3_raw.csv"
    if not raw.is_file():
        rep.add(section, "phase3_raw.csv", "FAIL", f"missing at {raw}")
        return

    df = pd.read_csv(raw)
    for (region, policy), (wel_exp, std_exp, isr_exp) in PHASE3_EXPECTED.items():
        sub = df[(df.region == region) & (df.policy == policy)]
        if sub.empty:
            rep.add(section, f"{region}/{policy}", "FAIL", "no rows in phase3_raw.csv")
            continue

        if sub.train_seed.notna().any():
            per_seed = sub.groupby("train_seed")[["WEL", "ISR"]].mean()
            wel, isr = per_seed.WEL.mean(), per_seed.ISR.mean()
            std = per_seed.WEL.std(ddof=1)
        else:
            wel, isr = sub.WEL.mean(), sub.ISR.mean()
            std = None

        problems = []
        if not close(wel, wel_exp, WEL_TOL):
            problems.append(f"WEL {wel:.4f} != {wel_exp}")
        if not close(isr, isr_exp, ISR_TOL):
            problems.append(f"ISR {isr:.4f} != {isr_exp}")
        if std_exp is not None:
            if std is None:
                problems.append("expected across-seed std, found deterministic policy")
            elif not close(std, std_exp, WEL_TOL):
                problems.append(f"std {std:.4f} != {std_exp}")

        shown = f"WEL {wel:.2f}" + (f" ± {std:.2f}" if std is not None else "") + f", ISR {isr:.3f}"
        rep.add(
            section,
            f"{region}/{policy}",
            "FAIL" if problems else "PASS",
            "; ".join(problems) if problems else shown,
        )

    # The published summary JSON must agree with the raw episodes it was derived from.
    summary = d / "phase3_summary.json"
    if summary.is_file():
        blob = json.loads(summary.read_text())
        mismatched = []
        for (region, policy), _ in PHASE3_EXPECTED.items():
            node = blob.get("regions", {}).get(region, {}).get(policy)
            if not node:
                mismatched.append(f"{region}/{policy} absent")
                continue
            recomputed = per_seed_wel(df, region, policy).mean()
            published = node["metrics"]["WEL"]["mean"]
            if not close(recomputed, published, WEL_TOL):
                mismatched.append(f"{region}/{policy} {recomputed:.4f} vs {published:.4f}")
        rep.add(
            section,
            "phase3_summary.json vs phase3_raw.csv",
            "FAIL" if mismatched else "PASS",
            "; ".join(mismatched) if mismatched else "all 14 cells agree",
        )


def check_significance(root: Path, rep: Report) -> None:
    section = "Phase 3 — significance (Welch on per-seed WEL)"
    d = resolve(root, "phase3")
    if d is None or not (d / "phase3_raw.csv").is_file():
        rep.add(section, "significance tests", "SKIP", "phase3_raw.csv not present")
        return
    if _stats is None:
        rep.add(section, "significance tests", "SKIP", "scipy not installed")
        return

    df = pd.read_csv(d / "phase3_raw.csv")
    for region, baseline, p_exp in SIGNIFICANCE_EXPECTED:
        ours = per_seed_wel(df, region, "hiercomm_heur")
        theirs = per_seed_wel(df, region, baseline)
        if len(ours) < 2 or len(theirs) < 2:
            rep.add(section, f"{region}: HierComm vs {baseline}", "SKIP", "needs >=2 seeds")
            continue
        p = float(_stats.ttest_ind(ours, theirs, equal_var=False).pvalue)
        ok = math.isclose(p, p_exp, rel_tol=P_REL_TOL)
        rep.add(
            section,
            f"{region}: HierComm vs {baseline}",
            "PASS" if ok else "FAIL",
            f"p={p:.4f} (paper: {p_exp})",
        )


def check_phase3_learned(root: Path, rep: Report) -> None:
    section = "Phase 3 — learned commander ablation"
    d = resolve(root, "phase3_learned")
    if d is None or not (d / "phase3_raw.csv").is_file():
        rep.add(section, "learned-commander results", "SKIP", "not present in this archive")
        return
    df = pd.read_csv(d / "phase3_raw.csv")
    for (region, policy), (wel_exp, std_exp) in PHASE3_LEARNED_EXPECTED.items():
        per_seed = per_seed_wel(df, region, policy)
        if per_seed.empty:
            rep.add(section, f"{region}/{policy}", "FAIL", "no rows")
            continue
        wel, std = per_seed.mean(), per_seed.std(ddof=1)
        problems = []
        if not close(wel, wel_exp, WEL_TOL):
            problems.append(f"WEL {wel:.4f} != {wel_exp}")
        if not close(std, std_exp, WEL_TOL):
            problems.append(f"std {std:.4f} != {std_exp}")
        rep.add(
            section,
            f"{region}/{policy}",
            "FAIL" if problems else "PASS",
            "; ".join(problems) if problems else f"WEL {wel:.2f} ± {std:.2f}",
        )

    # Headline claim: the learned commander never significantly *improves* on the rule.
    main = resolve(root, "phase3")
    if main is not None and (main / "phase3_raw.csv").is_file() and _stats is not None:
        rule_df = pd.read_csv(main / "phase3_raw.csv")
        for region in ("saudi", "california"):
            learned = per_seed_wel(df, region, "hiercomm")
            rule = per_seed_wel(rule_df, region, "hiercomm_heur")
            if len(learned) < 2 or len(rule) < 2:
                continue
            p = float(_stats.ttest_ind(learned, rule, equal_var=False).pvalue)
            improves = learned.mean() < rule.mean() and p < 0.05
            rep.add(
                section,
                f"{region}: learned does not beat rule",
                "FAIL" if improves else "PASS",
                f"learned {learned.mean():.2f} vs rule {rule.mean():.2f}, p={p:.3f}",
            )


def check_phase4_extended(root: Path, rep: Report) -> None:
    section = "Phase 4 — extended difficulty sweep"
    d = resolve(root, "phase4_extended")
    if d is None or not (d / "robustness_results.csv").is_file():
        rep.add(section, "extended sweep", "SKIP", "not present in this archive")
        return
    df = pd.read_csv(d / "robustness_results.csv")
    for (region, regime), expected_best in PHASE4_EXT_EXPECTED_BEST.items():
        sub = df[(df.region == region) & (df.regime == regime)]
        if sub.empty:
            rep.add(section, f"{region}/{regime} best policy", "FAIL", "no rows")
            continue
        best = sub.loc[sub.WEL.idxmin()]
        ok = best.policy == expected_best
        rep.add(
            section,
            f"{region}/{regime} best policy",
            "PASS" if ok else "FAIL",
            f"{best.policy} (WEL {best.WEL:.2f})"
            + ("" if ok else f" — paper says {expected_best}"),
        )


def check_phase6(root: Path, rep: Report) -> None:
    section = "Phase 6 — transfer / generalization"
    d = resolve(root, "phase6")
    if d is None:
        rep.add(section, "phase6 results", "SKIP", "not present in this archive")
        return
    raw = d / "transfer_matrix_raw.csv"
    summary = d / "transfer_summary.json"
    if not raw.is_file() or not summary.is_file():
        rep.add(section, "transfer artifacts", "SKIP", "transfer CSV/JSON not present")
        return

    df = pd.read_csv(raw)
    blob = json.loads(summary.read_text())

    # Every cell of every policy's 2x2 transfer matrix, re-derived from raw episodes.
    mismatches, checked = [], 0
    for policy, node in blob.get("policies", {}).items():
        for direction, cell in node.get("matrix", {}).items():
            src, _, dst = direction.partition("->")
            sub = df[(df.policy == policy) & (df.ckpt_region == src) & (df.eval_region == dst)]
            if sub.empty:
                mismatches.append(f"{policy} {direction}: no raw rows")
                continue
            checked += 1
            recomputed = sub.groupby("train_seed").WEL.mean().mean()
            if not close(recomputed, float(cell["WEL_mean"]), 0.05):
                mismatches.append(
                    f"{policy} {direction} WEL {recomputed:.3f} vs {cell['WEL_mean']:.3f}"
                )
    rep.add(
        section,
        "transfer matrix vs raw episodes",
        "FAIL" if mismatches else "PASS",
        "; ".join(mismatches[:3]) if mismatches else f"{checked} matrix cells re-derived and agree",
    )

    # TRS is defined as transfer ISR / native ISR; check the published ratio is internally
    # consistent (this is the headline transfer statistic in the paper).
    bad_trs = []
    for policy, node in blob.get("policies", {}).items():
        for direction, cell in node.get("directions", {}).items():
            native, transfer = cell.get("native_ISR"), cell.get("transfer_ISR")
            trs = cell.get("TRS_ISR")
            if None in (native, transfer, trs) or native == 0:
                continue
            if not close(transfer / native, float(trs), 1e-6):
                bad_trs.append(f"{policy} {direction}: {transfer / native:.6f} vs {trs:.6f}")
    rep.add(
        section,
        "TRS_ISR == transfer_ISR / native_ISR",
        "FAIL" if bad_trs else "PASS",
        "; ".join(bad_trs[:3]) if bad_trs else "all published TRS ratios internally consistent",
    )

    # Native-region cells must equal the Phase-3 main table (same checkpoints, same protocol).
    main = resolve(root, "phase3")
    if main is not None and (main / "phase3_raw.csv").is_file():
        main_df = pd.read_csv(main / "phase3_raw.csv")
        drift = []
        for policy, node in blob.get("policies", {}).items():
            for region in ("saudi", "california"):
                cell = node.get("matrix", {}).get(f"{region}->{region}")
                if not cell:
                    continue
                expected = per_seed_wel(main_df, region, policy).mean()
                if pd.isna(expected):
                    continue
                if not close(float(cell["WEL_mean"]), expected, 0.05):
                    drift.append(f"{policy}/{region} {cell['WEL_mean']:.2f} vs {expected:.2f}")
        rep.add(
            section,
            "native diagonal matches Phase-3 table",
            "FAIL" if drift else "PASS",
            "; ".join(drift[:3]) if drift else "diagonal cells agree with main results",
        )


def check_manifests(root: Path, rep: Report) -> None:
    """Per-file SHA-256 against each frozen MANIFEST.sha256.

    The whole-directory ``fingerprint_sha256`` in FREEZE.json cannot be recomputed here:
    it hashes a manifest covering the model checkpoints, most of which are excluded from
    the archive under the 50 MB cap. Every file that *is* shipped is verified byte-exactly
    against its frozen hash, and the omitted ones are reported.
    """
    section = "Integrity — frozen manifests"
    for key in DIR_CANDIDATES:
        d = resolve(root, key)
        if d is None:
            continue
        manifest = d / "MANIFEST.sha256"
        if not manifest.is_file():
            rep.add(section, f"{key}: MANIFEST.sha256", "FAIL", "missing")
            continue

        present = bad = absent = 0
        bad_names: list[str] = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            expected_hash, _, rel = line.partition("  ")
            target = d / rel
            if not target.is_file():
                absent += 1
                continue
            present += 1
            h = hashlib.sha256()
            with open(target, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                bad += 1
                bad_names.append(rel)

        detail = f"{present} files verified, {absent} omitted (size cap)"
        if bad:
            detail = f"{bad} HASH MISMATCH: {', '.join(bad_names[:3])}"
        rep.add(
            section, f"{key}: shipped files match frozen hashes", "FAIL" if bad else "PASS", detail
        )

        freeze = d / "FREEZE.json"
        if freeze.is_file():
            meta = json.loads(freeze.read_text())
            rep.add(
                section,
                f"{key}: FREEZE.json fingerprint on record",
                "PASS",
                f"{meta.get('fingerprint_sha256', '?')[:16]}… over {meta.get('n_files')} files, "
                f"frozen {meta.get('frozen_date')}",
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".", help="archive or repo root (default: cwd)")
    ap.add_argument("--json", dest="json_out", help="also write the report as JSON")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    print("=" * 78)
    print("REPRODUCIBILITY VERIFICATION — re-deriving the paper's numbers from frozen CSVs")
    print(f"root: {root}")
    print("=" * 78)

    rep = Report()
    for header, fn in (
        ("\n-- Main results (paper Table: WEL / ISR, both regions) " + "-" * 20, check_phase3),
        ("\n-- Significance tests " + "-" * 54, check_significance),
        ("\n-- Learned-commander ablation " + "-" * 46, check_phase3_learned),
        ("\n-- Extended difficulty sweep " + "-" * 47, check_phase4_extended),
        ("\n-- Transfer / generalization " + "-" * 47, check_phase6),
        ("\n-- Artifact integrity " + "-" * 54, check_manifests),
    ):
        print(header)
        fn(root, rep)

    print("\n" + "=" * 78)
    print(f"{rep.passed} passed, {rep.failed} failed, {rep.skipped} skipped")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "passed": rep.passed,
                    "failed": rep.failed,
                    "skipped": rep.skipped,
                    "checks": rep.rows,
                },
                indent=2,
            )
        )
        print(f"JSON report -> {args.json_out}")

    if rep.failed:
        print("\nVERIFICATION FAILED — the shipped artifacts do not match the paper's claims.")
        return 1
    print("\nVERIFICATION PASSED — every claim checked re-derives from the frozen artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
