"""Script to run/compile ablation studies and build report tables.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from wildfire_marl.ablation.groups import get_ablation_records


def main():
    records = get_ablation_records()
    df = pd.DataFrame(records)

    # Save tidy CSV
    csv_path = Path("results/runs/ablation_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved ablation summary CSV to {csv_path}")

    # Build report tables
    print("\n" + "=" * 65)
    print("AAAI ABLATION STUDY COMPARISONS")
    print("=" * 65)
    
    # 1. Architecture Ablation Table
    print("\nArchitecture Ablation (Saudi Region):")
    df_arch_s = df[(df["Region"] == "Saudi") & (df["Ablation Group"] == "Architecture")]
    print(df_arch_s[["Configuration", "Return", "Burned", "WEL", "CE"]].to_string(index=False))

    print("\nArchitecture Ablation (California Region):")
    df_arch_c = df[(df["Region"] == "California") & (df["Ablation Group"] == "Architecture")]
    print(df_arch_c[["Configuration", "Return", "Burned", "WEL", "CE"]].to_string(index=False))

    # 2. Cooperation Reward Ablation Table
    print("\nCooperative Coordination Ablation (Saudi Region):")
    df_coop = df[(df["Region"] == "Saudi") & (df["Ablation Group"] == "Cooperation")]
    print(df_coop[["Configuration", "Return", "Burned", "WEL", "CE"]].to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
