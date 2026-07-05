"""Generates publication-ready LaTeX tables for the AAAI paper submission.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from wildfire_marl.ablation.groups import get_ablation_records


def generate_latex_table(df: pd.DataFrame) -> str:
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\hline")
    latex.append(r"Configuration & Return & Burned Cells & WEL & CE \\")
    latex.append(r"\hline")
    
    for _, row in df.iterrows():
        latex.append(
            f"{row['Configuration']} & {row['Return']:.2f} & {row['Burned']:.1f} & "
            f"{row['WEL']:.1f} & {row['CE']:.2f} \\\\"
        )
        
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Ablation results.}")
    latex.append(r"\end{table}")
    return "\n".join(latex)


def main():
    records = get_ablation_records()
    df = pd.DataFrame(records)

    # 1. Saudi Architecture LaTeX
    saudi_arch = df[(df["Region"] == "Saudi") & (df["Ablation Group"] == "Architecture")]
    saudi_latex = generate_latex_table(saudi_arch)
    
    # 2. California Architecture LaTeX
    cali_arch = df[(df["Region"] == "California") & (df["Ablation Group"] == "Architecture")]
    cali_latex = generate_latex_table(cali_arch)

    # Write report files
    out_dir = Path("results/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "saudi_ablation_table.tex", "w") as f:
        f.write(saudi_latex)
    with open(out_dir / "cali_ablation_table.tex", "w") as f:
        f.write(cali_latex)

    print("LaTeX tables saved to results/runs/.")
    print("\nSaudi Architecture LaTeX Table:")
    print(saudi_latex)


if __name__ == "__main__":
    main()
