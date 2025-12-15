from __future__ import annotations

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _save_fig(fig, plots_dir: Path, name: str) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_contributions_heatmap(df: pd.DataFrame, plots_dir: Path) -> Path:
    contrib_df = pd.DataFrame(
        {
            "Noise contribution": df["noise_contrib"],
            "Zeroing contribution": df["zero_contrib"],
        },
        index=df["head"],
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        contrib_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Δ Accuracy (baseline − perturbed)"},
        ax=ax,
    )
    ax.set_title("Attention Head Contribution (Last Layer)")
    ax.set_ylabel("Head index")
    ax.set_xlabel("")
    return _save_fig(fig, plots_dir, "head_contribution_heatmap.png")


def plot_accuracy_heatmap(df: pd.DataFrame, baseline_acc: float, plots_dir: Path) -> Path:
    acc_df = pd.DataFrame(
        {
            "Baseline": [baseline_acc] * len(df),
            "Noise-perturbed": df["noise_mean_acc"],
            "Zeroed": df["zero_acc"],
        },
        index=df["head"],
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        acc_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_title("Accuracy per Head Perturbation")
    ax.set_ylabel("Head index")
    ax.set_xlabel("")
    return _save_fig(fig, plots_dir, "head_accuracy_heatmap.png")


def plot_noise_stats(df: pd.DataFrame, plots_dir: Path) -> Path:
    df_idx = df.set_index("head")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True, gridspec_kw={"width_ratios": [1, 1]})

    sns.heatmap(
        df_idx[["noise_mean_acc"]],
        ax=axes[0],
        cmap="viridis",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar=True,
        cbar_kws={"label": "Mean accuracy"},
    )
    axes[0].set_title("Noise Mean Accuracy")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Head index")

    sns.heatmap(
        df_idx[["noise_std_acc"]],
        ax=axes[1],
        cmap="magma",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar=True,
        cbar_kws={"label": "Std deviation"},
    )
    axes[1].set_title("Noise Std Deviation")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.suptitle("Noise Robustness of Attention Heads", y=1.02)
    return _save_fig(fig, plots_dir, "noise_stats_heatmap.png")


def plot_noise_contrib_ranking(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Bar chart ranking heads by noise contribution (baseline − noisy)."""
    ordered = df.sort_values(by="noise_contrib", ascending=False).copy()
    ordered["head_label"] = ordered["head"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=ordered,
        x="head_label",
        y="noise_contrib",
        palette="crest",
        order=ordered["head_label"].tolist(),
        ax=ax,
    )
    ax.set_title("Head Ranking by Noise Contribution")
    ax.set_xlabel("Head index (sorted by contribution)")
    ax.set_ylabel("Δ Accuracy (baseline − noisy)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    return _save_fig(fig, plots_dir, "head_noise_contrib_ranking.png")
