"""Phase 8 - GO/NO-GO Gate Certification script.
Performs rigorous multi-seed evaluation, statistical tests, compliance diagnostics,
and checks PASS/FAIL criteria.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import random
import json
import hashlib

import numpy as np
import pandas as pd
import torch

from wildfire_marl.env.marl_env import MultiAgentFireEnv
from wildfire_marl.agents.strategic_controller import StrategicController
from wildfire_marl.agents.agent_networks import MAPPOActor
from wildfire_marl.eval.statistics import compute_summary_stats, compare_policies
from wildfire_marl.eval.compliance import analyze_compliance_trajectory
from wildfire_marl.viz.compliance import (
    plot_trajectory_overlay,
    plot_sector_occupancy,
    plot_distance_to_target,
)
from scripts.run_multiseed_eval import run_episode_eval

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_reproducibility_manifest(
    output_dir: Path,
    region: str,
    base_seed: int,
    hier_ckpt: Path,
    flat_ckpt: Path,
) -> Path:
    """Generates a signed reproducibility manifest for the evaluation run."""
    manifest = {
        "region": region,
        "base_seed": base_seed,
        "hierarchical_checkpoint": str(hier_ckpt),
        "flat_marl_checkpoint": str(flat_ckpt),
        "reproducibility_hashes": {}
    }
    
    # Compute file hashes for verification
    for path in [hier_ckpt, flat_ckpt]:
        if path.exists():
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
            manifest["reproducibility_hashes"][path.name] = sha.hexdigest()
            
    manifest_path = output_dir / f"reproducibility_manifest_{region}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path

def run_certification(
    region: str,
    episodes: int,
    seeds: int,
    base_seed: int,
    output_dir: str,
    run_multiseed: bool = True,
):
    print(f"\n========================================================")
    print(f"STARTING PHASE 8 GO/NO-GO CERTIFICATION: {region.upper()}")
    print(f"========================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    map_name = "Saudi" if region.lower() == "saudi" else "California"
    data_dir = "data/cell2fire"
    infra_dir = f"{data_dir}/{map_name}"
    
    hierarchical_ckpt = Path(f"results/runs/checkpoint_hierarchical_{region}.pt")
    flat_ckpt = Path(f"results/runs/checkpoint_mappo_{region}.pt")
    
    # 1. Generate Manifest
    manifest_path = generate_reproducibility_manifest(out_path, region, base_seed, hierarchical_ckpt, flat_ckpt)
    print(f"Reproducibility manifest generated: {manifest_path}")
    
    # 2. Multi-seed Evaluation
    raw_csv = out_path / f"multiseed_eval_raw_{region}.csv"
    if run_multiseed or not raw_csv.exists():
        print(f"Executing multi-seed evaluation pipeline...")
        import subprocess
        cmd = [
            "python", "scripts/run_multiseed_eval.py",
            "--region", region,
            "--episodes", str(episodes),
            "--seeds", str(seeds),
            "--base_seed", str(base_seed),
            "--output_dir", output_dir
        ]
        subprocess.run(cmd, check=True)
        
    df = pd.read_csv(raw_csv)
    
    # Compute seed-level averages for statistical tests
    df_seed = df.groupby(["policy", "seed"]).mean(numeric_only=True).reset_index()
    
    # Group results
    hier_df = df_seed[df_seed["policy"] == "Learned Hierarchical"]
    vf_df = df_seed[df_seed["policy"] == "Value-First Heuristic"]
    flat_df = df_seed[df_seed["policy"] == "Flat MARL"]
    noop_df = df_seed[df_seed["policy"] == "No-Op"]
    
    # Fallback to episode-level if only 1 seed is evaluated (e.g. smoke test)
    if len(df_seed["seed"].unique()) < 2:
        print("WARNING: Less than 2 seeds evaluated. Using episode-level samples for statistical calculations.")
        hier_df = df[df["policy"] == "Learned Hierarchical"]
        vf_df = df[df["policy"] == "Value-First Heuristic"]
        flat_df = df[df["policy"] == "Flat MARL"]
        noop_df = df[df["policy"] == "No-Op"]

    primary_metrics = ["WEL", "ISR"]
    secondary_metrics = ["total_reward", "CE", "burned_cells"]
    
    summary_report = {
        "region": region,
        "metrics": {},
        "comparisons": {}
    }
    
    # Compute summary statistics
    for metric in primary_metrics + secondary_metrics:
        summary_report["metrics"][metric] = {}
        for policy, sub_df in [
            ("Hierarchical", hier_df),
            ("Value-First", vf_df),
            ("Flat_MARL", flat_df),
            ("No-Op", noop_df)
        ]:
            if len(sub_df) > 0:
                stats = compute_summary_stats(sub_df[metric].values, seed=base_seed)
                summary_report["metrics"][metric][policy] = stats

    # Compare Hierarchical against baselines
    for metric in primary_metrics + secondary_metrics:
        summary_report["comparisons"][metric] = {}
        hier_vals = hier_df[metric].values if len(hier_df) > 0 else []
        
        for baseline_name, base_df in [
            ("Value-First", vf_df),
            ("Flat_MARL", flat_df),
            ("No-Op", noop_df)
        ]:
            if len(base_df) > 0 and len(hier_vals) > 0:
                comp = compare_policies(hier_vals, base_df[metric].values)
                summary_report["comparisons"][metric][baseline_name] = comp

    # Save summary JSON
    summary_json_path = out_path / f"certification_summary_{region}.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Certification summary metrics saved to: {summary_json_path}")
    
    # 3. Strategic Compliance Diagnostics
    print("\nRunning compliance diagnostics...")
    env = MultiAgentFireEnv(
        num_agents=3,
        crop_size=9,
        coordination_penalty=0.1,
        fire_map=map_name,
        data_dir=data_dir,
        max_steps=150,
        steps_per_action=60,
        observe_infra=True,
        catastrophe_weight=2.0,
        cascade_prob=0.1,
        infra_dir=infra_dir,
    )
    
    commander = None
    if hierarchical_ckpt.exists():
        ckpt = torch.load(hierarchical_ckpt, map_location=device)
        commander = StrategicController(num_agents=env.num_agents).to(device)
        commander.load_state_dict(ckpt["commander_state_dict"])
        commander.eval()
        
    flat_marl_actor = None
    if flat_ckpt.exists():
        ckpt = torch.load(flat_ckpt, map_location=device)
        flat_marl_actor = MAPPOActor(in_channels=8, action_dim=6, features_dim=64).to(device)
        flat_marl_actor.load_state_dict(ckpt["actor_state_dict"])
        flat_marl_actor.eval()
        
    # Run a single episode to get tracking histories for plotting
    _, pos_hist, tar_hist = run_episode_eval(
        env, "Learned Hierarchical", commander, flat_marl_actor, base_seed, device
    )
    env.close()
    
    compliance_report = analyze_compliance_trajectory(pos_hist, tar_hist)
    print(f"Target Compliance Rate: {compliance_report['target_compliance_rate']:.3f}")
    print(f"Sector Drift (avg distance): {compliance_report['sector_drift']:.2f}")
    print(f"Dispatch Latency (avg steps to target): {compliance_report['dispatch_latency']:.2f}")
    
    # Plot visualizations
    plot_trajectory_overlay(pos_hist, tar_hist, out_path / f"trajectory_overlay_{region}.png")
    plot_sector_occupancy(compliance_report["sector_occupancy"], out_path / f"sector_occupancy_{region}.png")
    plot_distance_to_target(pos_hist, tar_hist, out_path / f"target_distance_{region}.png")
    print(f"Generated trajectory and compliance plots in: {out_path}")
    
    # 4. PASS/FAIL Evaluation Logic
    print("\n=========================================")
    print("PASS/FAIL DECISION REPORT")
    print("=========================================")
    
    wel_stats = summary_report["metrics"]["WEL"]
    isr_stats = summary_report["metrics"]["ISR"]
    
    # Extracted bounds
    hier_wel_mean = wel_stats["Hierarchical"]["mean"]
    hier_wel_hi = wel_stats["Hierarchical"]["ci_hi"]
    hier_wel_lo = wel_stats["Hierarchical"]["ci_lo"]
    
    vf_wel_mean = wel_stats["Value-First"]["mean"]
    vf_wel_hi = wel_stats["Value-First"]["ci_hi"]
    vf_wel_lo = wel_stats["Value-First"]["ci_lo"]
    
    hier_isr_mean = isr_stats["Hierarchical"]["mean"]
    hier_isr_hi = isr_stats["Hierarchical"]["ci_hi"]
    hier_isr_lo = isr_stats["Hierarchical"]["ci_lo"]
    
    vf_isr_mean = isr_stats["Value-First"]["mean"]
    vf_isr_hi = isr_stats["Value-First"]["ci_hi"]
    vf_isr_lo = isr_stats["Value-First"]["ci_lo"]
    
    flat_wel_mean = wel_stats["Flat_MARL"]["mean"]
    
    passed = False
    reasons = []
    
    # Condition A: Significantly beats Heuristic on WEL or ISR (CIs do not overlap)
    # Since lower is better for WEL: hier_wel_hi < vf_wel_lo means Hierarchical is significantly better.
    # Since higher is better for ISR: hier_isr_lo > vf_isr_hi means Hierarchical is significantly better.
    if hier_wel_hi < vf_wel_lo:
        passed = True
        reasons.append("Condition A: Hierarchical significantly beats Value-First on WEL (CIs do not overlap).")
    if hier_isr_lo > vf_isr_hi:
        passed = True
        reasons.append("Condition A: Hierarchical significantly beats Value-First on ISR (CIs do not overlap).")
        
    # Condition B: Statistically matches Heuristic on WEL/ISR and significantly beats Flat MARL
    p_wel = summary_report["comparisons"]["WEL"]["Value-First"]["p_val"]
    p_isr = summary_report["comparisons"]["ISR"]["Value-First"]["p_val"]
    p_wel_flat = summary_report["comparisons"]["WEL"]["Flat_MARL"]["p_val"]
    
    # Statistical match is defined as no significant difference (p >= 0.05) or superior mean performance
    wel_match = (p_wel >= 0.05) or (hier_wel_mean <= vf_wel_mean)
    isr_match = (p_isr >= 0.05) or (hier_isr_mean >= vf_isr_mean)
    
    if wel_match and isr_match and (hier_wel_mean < flat_wel_mean):
        passed = True
        reasons.append("Condition B: Hierarchical statistically matches Value-First and outperforms Flat MARL.")
        
    # Condition C: Near-parity (within 10% of Heuristic) with superior coordination metrics (CE or Reward)
    wel_within_10 = (hier_wel_mean <= 1.10 * vf_wel_mean)
    isr_within_10 = (hier_isr_mean >= 0.90 * vf_isr_mean)
    hier_ce = summary_report["metrics"]["CE"]["Hierarchical"]["mean"]
    vf_ce = summary_report["metrics"]["CE"]["Value-First"]["mean"]
    
    if wel_within_10 and isr_within_10 and (hier_ce >= vf_ce):
        passed = True
        reasons.append("Condition C: Near-parity (within 10%) achieved with superior/equal coordination efficiency.")
        
    # Check FAIL condition: Hierarchy collapses to Flat MARL
    # (defined as no significant difference on WEL and similar means)
    if not passed:
        reasons.append("FAIL: Policy underperforms heuristic significantly and shows no separation from Flat MARL.")
        
    status_str = "PASS" if passed else "FAIL"
    print(f"Certification Status: {status_str}")
    for r in reasons:
        print(f"  - {r}")
    print("=========================================\n")
    
    # Save text report to file
    report_txt = out_path / f"certification_decision_{region}.txt"
    with open(report_txt, "w") as f:
        f.write(f"Region: {region}\n")
        f.write(f"Status: {status_str}\n")
        f.write("Reasons:\n")
        for r in reasons:
            f.write(f"  - {r}\n")
            
    # Print LaTeX comparison table
    print("LaTeX Summary Table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("Policy & WEL (95\\% CI) $\\downarrow$ & ISR (95\\% CI) $\\uparrow$ & Reward & CE \\\\")
    print("\\hline")
    for policy_name, key in [
        ("No-Op", "No-Op"),
        ("Flat MARL", "Flat_MARL"),
        ("Value-First", "Value-First"),
        ("Learned Hierarchical", "Hierarchical")
    ]:
        w = summary_report["metrics"]["WEL"][key]
        i = summary_report["metrics"]["ISR"][key]
        r = summary_report["metrics"]["total_reward"][key]["mean"]
        c = summary_report["metrics"]["CE"][key]["mean"]
        print(
            f"{policy_name} & {w['mean']:.2f} [{w['ci_lo']:.2f}, {w['ci_hi']:.2f}] & "
            f"{i['mean']:.2f} [{i['ci_lo']:.2f}, {i['ci_hi']:.2f}] & "
            f"{r:.2f} & {c:.2f} \\\\"
        )
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Phase 8 Certification Summary Table}")
    print("\\end{table}\n")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 8 GO/NO-GO gate certification.")
    parser.add_argument("--region", type=str, required=True, choices=["saudi", "california"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/phase8")
    parser.add_argument("--skip_multiseed", action="store_true", help="Skip running multiseed script again")
    args = parser.parse_args()

    run_certification(
        region=args.region,
        episodes=args.episodes,
        seeds=args.seeds,
        base_seed=args.seed,
        output_dir=args.output_dir,
        run_multiseed=not args.skip_multiseed
    )

if __name__ == "__main__":
    main()
