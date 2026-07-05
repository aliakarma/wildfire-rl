"""Consolidated literature benchmarking script comparing our methods against baselines."""

from __future__ import annotations

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run literature benchmark comparisons.")
    parser.add_argument("--episodes", type=int, default=15, help="Number of comparison episodes")
    args = parser.parse_args()

    print("Running literature benchmarks...")

    # Run hierarchical evaluation for Saudi
    print("\n--- Running Saudi benchmarks ---")
    subprocess.run(
        [
            "python",
            "scripts/eval_hierarchical.py",
            "--region",
            "saudi",
            "--episodes",
            str(args.episodes),
        ],
        check=True,
    )

    # Run hierarchical evaluation for California
    print("\n--- Running California benchmarks ---")
    subprocess.run(
        [
            "python",
            "scripts/eval_hierarchical.py",
            "--region",
            "california",
            "--episodes",
            str(args.episodes),
        ],
        check=True,
    )

    print("\nLiterature benchmarking completed successfully.")


if __name__ == "__main__":
    main()
