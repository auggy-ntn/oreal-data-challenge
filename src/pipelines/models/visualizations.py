"""Visualization module for L'Oréal MMM.

This module creates all visualizations for the MMM analysis:
- Saturation curves by channel
- Time series with model fit overlay
- Media contribution charts
- Combined dashboard visualizations
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants.paths import PROJECT_ROOT
from src.pipelines.models.transformations import (
    get_default_params,
    get_saturation_curve_points,
    hill_saturation,
)
from src.utils.logger import logger

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_saturation_curves_plot(
    channels: list[str] | None = None,
) -> Path:
    """Create saturation curves for all channels.

    Args:
        channels: List of channels to plot (None = all)

    Returns:
        Path to saved plot
    """
    if channels is None:
        channels = [
            "linear",
            "bvod",
            "meta",
            "tik_tok",
            "pinterest",
            "youtube",
            "google",
            "amazon",
            "tesco",
        ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    x = np.linspace(0, 2, 100)

    for i, channel in enumerate(channels[:9]):
        ax = axes[i]
        params = get_default_params(channel)
        K = params["saturation_k"]
        S = params["saturation_s"]

        # Plot curve
        y = hill_saturation(x, K, S)
        ax.plot(x, y, "b-", linewidth=2, label=f"K={K}, S={S}")

        # Mark ABCD points
        abcd = get_saturation_curve_points(K, S)
        for point_name, (px, py) in abcd.items():
            if px <= 2:  # Only plot if in range
                ax.scatter([px], [py], s=80, zorder=5)
                ax.annotate(
                    point_name,
                    (px, py),
                    fontsize=10,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Normalized Execution")
        ax.set_ylabel("Saturated Effect")
        ax.set_title(channel.replace("_", " ").title(), fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(
            y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% saturation"
        )

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "saturation_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(f"Saved saturation curves to {plot_path}")
    return plot_path


def create_model_fit_plot(
    df: pd.DataFrame,
    y_actual: pd.Series,
    y_pred: np.ndarray,
    title: str,
) -> Path:
    """Create time series plot with model fit overlay.

    Args:
        df: DataFrame with 'week' column
        y_actual: Actual values
        y_pred: Predicted values
        title: Plot title

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df["week"], y_actual, "b-", linewidth=1.5, label="Actual", alpha=0.8)
    ax.fill_between(df["week"], y_actual, alpha=0.2)
    ax.plot(df["week"], y_pred, "r--", linewidth=2, label="Predicted")

    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"model_fit_{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def create_contribution_pie_chart(
    due_to_df: pd.DataFrame,
    title: str,
) -> Path:
    """Create pie chart of channel contributions.

    Args:
        due_to_df: Due-To attribution results
        title: Chart title

    Returns:
        Path to saved chart
    """
    # Filter to positive contributions
    positive_df = due_to_df[due_to_df["due_to_units"] > 0].copy()

    if len(positive_df) == 0:
        logger.warning("No positive contributions to plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Top 8 channels + "Other"
    if len(positive_df) > 8:
        top8 = positive_df.nlargest(8, "due_to_units")
        other = positive_df.nsmallest(len(positive_df) - 8, "due_to_units")
        other_sum = other["due_to_units"].sum()

        labels = list(top8["channel"]) + ["Other"]
        sizes = list(top8["due_to_units"]) + [other_sum]
    else:
        labels = list(positive_df["channel"])
        sizes = list(positive_df["due_to_units"])

    # Clean labels
    labels = [label.replace("_", " ").title() for label in labels]

    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        pctdistance=0.8,
        startangle=90,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"contribution_pie_{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def create_roi_bar_chart(
    roi_df: pd.DataFrame,
    title: str,
) -> Path:
    """Create horizontal bar chart of ROI by channel.

    Args:
        roi_df: ROI analysis results
        title: Chart title

    Returns:
        Path to saved chart
    """
    # Filter to valid ROI
    valid_df = roi_df[~roi_df["roi"].isna()].copy()
    valid_df = valid_df.sort_values("roi", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [
        "green" if r > 1 else "orange" if r > 0 else "red" for r in valid_df["roi"]
    ]

    y_pos = range(len(valid_df))
    ax.barh(y_pos, valid_df["roi"], color=colors, edgecolor="black", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_", " ").title() for c in valid_df["channel"]])
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(
        x=1,
        color="blue",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Break-even (ROI=1)",
    )
    ax.set_xlabel("ROI (Incremental Revenue / Investment)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"roi_bar_{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def create_investment_vs_roi_scatter(
    roi_df: pd.DataFrame,
    title: str,
) -> Path:
    """Create scatter plot of Investment vs ROI.

    Args:
        roi_df: ROI analysis results
        title: Chart title

    Returns:
        Path to saved chart
    """
    valid_df = roi_df[(~roi_df["roi"].isna()) & (roi_df["investment"] > 0)].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    sizes = (valid_df["investment"] / valid_df["investment"].max()) * 500 + 50
    colors = [
        "green" if r > 1 else "orange" if r > 0 else "red" for r in valid_df["roi"]
    ]

    ax.scatter(
        valid_df["investment"] / 1e6,
        valid_df["roi"],
        s=sizes,
        c=colors,
        alpha=0.6,
        edgecolors="black",
    )

    # Add labels
    for _, row in valid_df.iterrows():
        ax.annotate(
            row["channel"].replace("_", " ").title(),
            (row["investment"] / 1e6, row["roi"]),
            fontsize=8,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="blue", linestyle="--", alpha=0.5, label="Break-even")
    ax.set_xlabel("Investment (£M)")
    ax.set_ylabel("ROI")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"investment_roi_{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def generate_all_visualizations(
    model_results: dict,
    insights_results: dict,
) -> list[Path]:
    """Generate all visualizations for the MMM analysis.

    Args:
        model_results: Results from build_and_evaluate_models()
        insights_results: Results from run_full_insights_pipeline()

    Returns:
        List of paths to generated plots
    """
    logger.info("Generating all visualizations...")
    plots = []

    # Saturation curves
    plots.append(create_saturation_curves_plot())

    # Model fit plots
    df = model_results["df"]

    offline_fit = create_model_fit_plot(
        df,
        model_results["offline_y"],
        model_results["offline_model"].fittedvalues,
        "Offline Units Model Fit",
    )
    plots.append(offline_fit)

    online_fit = create_model_fit_plot(
        df,
        model_results["online_y"],
        model_results["online_model"].fittedvalues,
        "Online Units Model Fit",
    )
    plots.append(online_fit)

    # Contribution pie charts
    if "offline_due_to" in insights_results:
        pie_path = create_contribution_pie_chart(
            insights_results["offline_due_to"],
            "Offline Contribution",
        )
        if pie_path:
            plots.append(pie_path)

    if "online_due_to" in insights_results:
        pie_path = create_contribution_pie_chart(
            insights_results["online_due_to"],
            "Online Contribution",
        )
        if pie_path:
            plots.append(pie_path)

    # ROI bar charts
    if "offline_roi" in insights_results:
        plots.append(
            create_roi_bar_chart(
                insights_results["offline_roi"],
                "Offline ROI",
            )
        )

    if "online_roi" in insights_results:
        plots.append(
            create_roi_bar_chart(
                insights_results["online_roi"],
                "Online ROI",
            )
        )

    logger.success(f"Generated {len(plots)} visualization plots")
    return plots


if __name__ == "__main__":
    # Generate saturation curves standalone
    create_saturation_curves_plot()
