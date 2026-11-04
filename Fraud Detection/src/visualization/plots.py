"""
Visualization utilities for fraud detection analysis.
Generates publication-quality charts for class-imbalance distributions,
supporting dual linear/log scales to clearly visualize severe minority skews.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.data.validator import ImbalanceMetrics


def plot_class_distribution(
    metrics: ImbalanceMetrics,
    dataset_name: str,
    output_path: Optional[Path] = None,
    sample_info: Optional[str] = None,
) -> Path:
    """
    Renders and saves a two-panel publication-grade figure of class distribution:
    - Panel 1: Bar chart comparing Legitimate vs Fraudulent counts (linear & log-scaled).
    - Panel 2: Normalized proportion chart illustrating extreme class imbalance.

    Args:
        metrics: ImbalanceMetrics containing counts and percentages.
        dataset_name: Display title for the dataset.
        output_path: Destination file path for PNG image.
        sample_info: Explanatory text if data was sampled (preserves transparency).

    Returns:
        Path to the saved figure file.
    """
    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_path = (
            project_root
            / "outputs"
            / "figures"
            / f"{dataset_name.lower().replace(' ', '_')}_class_distribution.png"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Style and palette setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    classes = ["Legitimate (Class 0)", "Fraudulent (Class 1)"]
    counts = [metrics.legitimate_count, metrics.fraud_count]
    percentages = [metrics.legitimate_percentage, metrics.fraud_percentage]
    colors = ["#2b5c8f", "#d9534f"]

    # -------------------------------------------------------------
    # Panel 1: Count Bar Chart with Log-Scale to reveal minority
    # -------------------------------------------------------------
    bars = ax1.bar(classes, counts, color=colors, width=0.55, edgecolor="black", linewidth=1.1, alpha=0.9)
    ax1.set_yscale("log")
    ax1.set_ylabel("Transaction Count (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Class Frequencies (Log Scale)\n{dataset_name}", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate counts and percentages on top of bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.annotate(
            f"{count:,}\n({pct:.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add ratio annotation box
    ratio_text = (
        f"Total: {metrics.total_samples:,}\n"
        f"Imbalance Ratio: {metrics.imbalance_ratio:,.1f} : 1\n"
        f"Fraud Rate: {metrics.fraud_percentage:.4f}%\n"
        + (f"Note: {sample_info}" if sample_info else "Scope: Full Dataset (No Truncation)")
    )
    ax1.text(
        0.05,
        0.95,
        ratio_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
    )

    # -------------------------------------------------------------
    # Panel 2: Donut Proportion Chart showing extreme disparity
    # -------------------------------------------------------------
    # For extreme imbalance where fraud slice would be invisible in a standard pie,
    # we represent a stacked proportion bar or custom donut with exploded slice
    wedges, texts, autotexts = ax2.pie(
        [max(metrics.legitimate_percentage, 0.001), max(metrics.fraud_percentage, 0.001)],
        labels=["Legitimate", "Fraud"],
        autopct="%1.2f%%",
        pctdistance=0.75,
        colors=colors,
        explode=(0, 0.15),
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.1),
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(10)
        t.set_fontweight("bold")

    ax2.set_title("Relative Class Proportion", fontsize=12, fontweight="bold", pad=12)

    # Overall figure title
    plt.suptitle(
        f"Comparative Analysis: Class-Imbalance Profile\n{dataset_name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path

