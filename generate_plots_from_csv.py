from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from plotting_utils import (
    plot_contributions_heatmap,
    plot_accuracy_heatmap,
    plot_noise_stats,
    plot_noise_contrib_ranking,
)


def infer_baseline(df: pd.DataFrame) -> float:
    """Infer baseline accuracy from stored contributions."""
    noise_est = df["noise_mean_acc"] + df["noise_contrib"]
    zero_est = df["zero_acc"] + df["zero_contrib"]
    candidates = pd.concat([noise_est, zero_est], ignore_index=True)
    candidates = candidates.dropna()
    if candidates.empty:
        raise ValueError("Cannot infer baseline: no valid contribution columns.")
    return float(candidates.mean())


def main():
    parser = argparse.ArgumentParser(description="Generate plots from an existing head perturbation CSV.")
    parser.add_argument("--csv", dest="csv_path", type=Path, default=Path("head_perturbation_outproj_results.csv"), help="Path to results CSV")
    parser.add_argument("--plots-dir", dest="plots_dir", type=Path, default=Path("plots"), help="Directory to save plots")
    parser.add_argument("--baseline", dest="baseline", type=float, default=0.6421, help="Optional baseline accuracy override")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "head" not in df.columns:
        raise ValueError("CSV must contain a 'head' column.")

    baseline_acc = args.baseline if args.baseline is not None else infer_baseline(df)

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = [
        plot_contributions_heatmap(df, args.plots_dir),
        plot_accuracy_heatmap(df, baseline_acc, args.plots_dir),
        plot_noise_stats(df, args.plots_dir),
        plot_noise_contrib_ranking(df, args.plots_dir),
    ]

    print(f"Baseline used for plots: {baseline_acc:.4f}")
    for p in plot_paths:
        print(f"Saved plot -> {p}")


if __name__ == "__main__":
    main()
