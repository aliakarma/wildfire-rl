"""Phase 6 (peer-review remediation): gather compute + release provenance.

Writes results/phase6_peer/{sysinfo.json, checkpoint_hashes.json, release_manifest.json}:
system/library versions, SHA-256 of every REPORTED policy checkpoint, and a release
manifest (checkpoints + configs + data + key result artifacts).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

OUT = Path("results/phase6_peer")
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- system / library versions
libs = {}
for pkg in ["numpy", "torch", "gymnasium", "pettingzoo", "pandas", "scipy", "matplotlib"]:
    try:
        libs[pkg] = version(pkg)
    except Exception:
        libs[pkg] = "n/a"

sysinfo = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "processor": platform.processor() or "n/a",
    "machine": platform.machine(),
    "libraries": libs,
    "device": "CPU (no CUDA used for any peer-remediation run)",
    "seeds": {
        "training_seed": 42,
        "evaluation_seeds": [42, 1042, 2042, 3042, 4042],
        "episodes_per_seed": 15,
        "bc_demo_seed_base": 5000,
    },
}
(OUT / "sysinfo.json").write_text(json.dumps(sysinfo, indent=2))

# --- checkpoint hashes for REPORTED policies only
# Reported learned policies: hierarchical (shipped), and the Phase-2 rehabilitated
# flat MAPPO / QMIX / CommNet. Non-learned policies (No-Op, Value-First, Greedy-Risk,
# Local Reactive) have no checkpoint.
reported = {
    "checkpoint_hierarchical_saudi.pt": "results/runs/checkpoint_hierarchical_saudi.pt",
    "checkpoint_hierarchical_california.pt": "results/runs/checkpoint_hierarchical_california.pt",
    "checkpoint_mappo_saudi.pt": "results/phase2_peer/checkpoint_mappo_saudi.pt",
    "checkpoint_mappo_california.pt": "results/phase2_peer/checkpoint_mappo_california.pt",
    "checkpoint_qmix_saudi.pt": "results/phase2_peer/checkpoint_qmix_saudi.pt",
    "checkpoint_qmix_california.pt": "results/phase2_peer/checkpoint_qmix_california.pt",
    "checkpoint_commnet_saudi.pt": "results/phase2_peer/checkpoint_commnet_saudi.pt",
    "checkpoint_commnet_california.pt": "results/phase2_peer/checkpoint_commnet_california.pt",
}
hashes = {}
for name, rel in reported.items():
    p = Path(rel)
    hashes[name] = {
        "path": rel,
        "sha256": sha256(p) if p.exists() else "MISSING",
        "sha256_16": (sha256(p)[:16] if p.exists() else "MISSING"),
    }
(OUT / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2))

# --- release manifest: hash key result artifacts + configs + data
manifest_targets = [
    "results/phase2_peer/multiseed_eval_raw_saudi.csv",
    "results/phase2_peer/multiseed_eval_raw_california.csv",
    "results/phase2_peer/eval_summary_saudi.json",
    "results/phase2_peer/eval_summary_california.json",
    "results/phase3_peer/ablation_aggregate_saudi.csv",
    "results/phase3_peer/ablation_summary_saudi.json",
    "results/phase4_peer/tost_results.json",
    "results/phase4_peer/stats_tables.json",
    "results/phase5_peer/generalization_summary_saudi.json",
    "results/phase5_peer/generalization_summary_california.json",
    "results/phase5_peer/scale_summary_saudi.json",
    "results/phase5_peer/scale_summary_california.json",
    "results/data_manifest.json",
]
manifest = {"checkpoints": hashes, "artifacts": {}}
for rel in manifest_targets:
    p = Path(rel)
    if p.exists():
        manifest["artifacts"][rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
(OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2))

print("Wrote:")
print(" ", OUT / "sysinfo.json")
print(" ", OUT / "checkpoint_hashes.json", f"({len(hashes)} checkpoints)")
print(" ", OUT / "release_manifest.json", f"({len(manifest['artifacts'])} artifacts)")
print("\nLibraries:", json.dumps(libs))
print("\nAppendix D hash table (first 16 chars):")
for name, h in hashes.items():
    print(f"  {name:40s} {h['sha256_16']}")
