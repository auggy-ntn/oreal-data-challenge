"""Insights extraction and optimization module for L'Oréal MMM.

This module calculates:
- Incremental attribution ("Due-To" sales by channel)
- ROI for each A&P touchpoint
- Saturation analysis (ABCD points)
- Budget optimization recommendations
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants.paths import PROJECT_ROOT
from src.pipelines.models.mmm_model import (
    build_and_evaluate_models,
    extract_ap_coefficients,
)
from src.pipelines.models.transformations import (
    get_default_params,
    get_saturation_curve_points,
)
from src.utils.logger import logger

warnings.filterwarnings("ignore")

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_due_to_sales(
    model_results: dict,
    channel: str = "offline",
) -> pd.DataFrame:
    """Calculate incremental 'Due-To' sales by channel.

    Formula: Due-To = coefficient × sum(transformed_execution)

    Args:
        model_results: Results from build_and_evaluate_models()
        channel: 'offline' or 'online'

    Returns:
        DataFrame with channel, due_to_units, percent_contribution
    """
    model = model_results[f"{channel}_model"]
    df = model_results["df"]

    # Get A&P coefficients
    ap_coefs = extract_ap_coefficients(model)
    if ap_coefs.empty:
        return pd.DataFrame()

    due_to_data = []

    for _, row in ap_coefs.iterrows():
        channel_name = row["channel"]
        coef = row["coefficient"]
        transformed_col = f"{channel_name}_transformed"

        if transformed_col in df.columns:
            # Sum of transformed values × coefficient = incremental units
            total_transformed = df[transformed_col].sum()
            due_to_units = coef * total_transformed

            due_to_data.append(
                {
                    "channel": channel_name,
                    "coefficient": coef,
                    "due_to_units": due_to_units,
                    "significant": row["significant"],
                }
            )

    result = pd.DataFrame(due_to_data)

    if len(result) > 0:
        # Calculate percentage contribution
        total_positive = result[result["due_to_units"] > 0]["due_to_units"].sum()
        result["pct_contribution"] = (
            result["due_to_units"].clip(lower=0) / total_positive * 100
        ).fillna(0)
        result = result.sort_values("due_to_units", ascending=False).reset_index(
            drop=True
        )

    return result


def calculate_roi(
    due_to_df: pd.DataFrame,
    df: pd.DataFrame,
    avg_price: float,
) -> pd.DataFrame:
    """Calculate ROI for each channel.

    Formula: ROI = (incremental_units × avg_price) / investment

    Args:
        due_to_df: Due-To attribution results
        df: Unified dataset with investment data
        avg_price: Average selling price

    Returns:
        DataFrame with ROI metrics
    """
    roi_data = []

    for _, row in due_to_df.iterrows():
        channel = row["channel"]
        due_to_units = row["due_to_units"]

        # Get total investment for this channel
        investment_col = f"{channel}_investment"
        if investment_col in df.columns:
            total_investment = df[investment_col].sum()
        else:
            total_investment = 0

        # Calculate ROI
        incremental_revenue = due_to_units * avg_price

        if total_investment > 0:
            roi = incremental_revenue / total_investment
        else:
            roi = np.nan

        roi_data.append(
            {
                "channel": channel,
                "due_to_units": due_to_units,
                "incremental_revenue": incremental_revenue,
                "investment": total_investment,
                "roi": roi,
                "significant": row["significant"],
            }
        )

    result = pd.DataFrame(roi_data)
    result = result.sort_values("roi", ascending=False, na_position="last")

    return result


def get_saturation_status(
    df: pd.DataFrame,
    channel: str,
) -> dict:
    """Determine saturation status for a channel.

    Args:
        df: Unified dataset
        channel: Channel name

    Returns:
        Dictionary with saturation metrics and status
    """
    params = get_default_params(channel)
    abcd = get_saturation_curve_points(
        K=params["saturation_k"], S=params["saturation_s"]
    )

    # Get current execution level
    execution_col = f"{channel}_execution"
    if execution_col not in df.columns:
        return {}

    # Normalize execution
    max_exec = df[execution_col].max()
    if max_exec == 0:
        return {"status": "no_spend", "current_level": 0, "saturation_pct": 0}

    current_normalized = df[execution_col].mean() / max_exec

    # Determine status based on ABCD points
    if current_normalized < abcd["A"][0]:
        status = "below_threshold"
        saturation_pct = 5
    elif current_normalized < abcd["B"][0]:
        status = "low_saturation"
        saturation_pct = 25
    elif current_normalized < abcd["C"][0]:
        status = "moderate_saturation"
        saturation_pct = 50
    elif current_normalized < abcd["D"][0]:
        status = "high_saturation"
        saturation_pct = 75
    else:
        status = "fully_saturated"
        saturation_pct = 95

    return {
        "status": status,
        "current_level": current_normalized,
        "saturation_pct": saturation_pct,
        "abcd_points": abcd,
    }


def create_bubble_chart(
    roi_df: pd.DataFrame,
    title: str,
) -> Path:
    """Create bubble chart of ROI vs Contribution.

    Bubble size = investment
    X-axis = ROI
    Y-axis = Due-To Units (contribution)
    Color = significance

    Args:
        roi_df: ROI analysis results
        title: Chart title

    Returns:
        Path to saved chart
    """
    # Filter to valid data
    plot_df = roi_df[roi_df["investment"] > 0].copy()

    if len(plot_df) == 0:
        logger.warning("No data to plot in bubble chart")
        return None

    fig, ax = plt.subplots(figsize=(14, 10))

    # Normalize bubble sizes
    max_investment = plot_df["investment"].max()
    sizes = (plot_df["investment"] / max_investment) * 2000 + 100

    # Colors based on significance and ROI sign
    colors = []
    for _, row in plot_df.iterrows():
        if row["roi"] > 1 and row["significant"]:
            colors.append("green")  # High ROI, significant
        elif row["roi"] > 0 and row["significant"]:
            colors.append("lightgreen")  # Positive ROI, significant
        elif row["roi"] > 0:
            colors.append("gray")  # Positive but not significant
        else:
            colors.append("red")  # Negative ROI

    ax.scatter(
        plot_df["roi"],
        plot_df["due_to_units"],
        s=sizes,
        c=colors,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
    )

    # Add labels
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["channel"].replace("_", " ").title(),
            (row["roi"], row["due_to_units"]),
            fontsize=9,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1, color="blue", linestyle="--", alpha=0.3, label="ROI = 1x")

    ax.set_xlabel("ROI (Incremental Revenue / Investment)", fontsize=12)
    ax.set_ylabel("Due-To Units (Incremental Sales)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="High ROI + Significant"),
        Patch(
            facecolor="lightgreen",
            edgecolor="black",
            label="Positive ROI + Significant",
        ),
        Patch(facecolor="gray", edgecolor="black", label="Positive ROI (n.s.)"),
        Patch(facecolor="red", edgecolor="black", label="Negative ROI"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"bubble_chart_{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def create_waterfall_chart(
    due_to_df: pd.DataFrame,
    base_sales: float,
    total_sales: float,
    title: str,
) -> Path:
    """Create waterfall chart showing sales decomposition.

    Args:
        due_to_df: Due-To attribution results
        base_sales: Baseline sales (intercept + controls)
        total_sales: Total actual sales
        title: Chart title

    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter significant channels with meaningful contribution
    sig_df = due_to_df[due_to_df["significant"]].copy()
    sig_df = sig_df[sig_df["due_to_units"].abs() > total_sales * 0.001]  # >0.1%
    sig_df = sig_df.sort_values("due_to_units", ascending=False)

    # Prepare waterfall data
    categories = ["Baseline"] + list(sig_df["channel"]) + ["Total"]

    # Calculate running totals
    running_total = [base_sales]
    for val in sig_df["due_to_units"]:
        running_total.append(running_total[-1] + val)
    running_total.append(total_sales)

    # Create bars
    bar_starts = [0] + running_total[:-2] + [0]
    bar_heights = [base_sales] + list(sig_df["due_to_units"]) + [running_total[-2]]

    colors = (
        ["steelblue"]
        + ["green" if v > 0 else "red" for v in sig_df["due_to_units"]]
        + ["steelblue"]
    )

    bars = ax.bar(
        range(len(categories)),
        bar_heights,
        bottom=bar_starts,
        color=colors,
        edgecolor="black",
    )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Units")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, bar_heights, strict=True)):
        if i == 0 or i == len(bars) - 1:
            label_pos = bar.get_height() / 2 + bar.get_y()
            ax.annotate(
                f"{val:,.0f}",
                (bar.get_x() + bar.get_width() / 2, label_pos),
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        else:
            label_pos = bar.get_y() + bar.get_height() + (1000 if val > 0 else -3000)
            ax.annotate(
                f"{val:+,.0f}",
                (bar.get_x() + bar.get_width() / 2, label_pos),
                ha="center",
                va="bottom" if val > 0 else "top",
                fontsize=8,
            )

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"waterfall_{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def generate_optimization_recommendations(
    roi_df: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate budget optimization recommendations.

    Args:
        roi_df: ROI analysis results
        df: Unified dataset

    Returns:
        DataFrame with recommendations
    """
    recommendations = []

    for _, row in roi_df.iterrows():
        channel = row["channel"]
        roi = row["roi"]
        investment = row["investment"]
        significant = row["significant"]

        # Get saturation status
        sat_status = get_saturation_status(df, channel)
        saturation = sat_status.get("status", "unknown")

        # Determine recommendation
        if not significant:
            recommendation = "Monitor"
            action = "Continue monitoring; effect not statistically significant"
        elif roi > 2 and saturation in ["low_saturation", "below_threshold"]:
            recommendation = "Significantly Increase"
            action = "High ROI with room for growth; priority investment channel"
        elif roi > 1 and saturation == "moderate_saturation":
            recommendation = "Moderate Increase"
            action = "Good ROI; consider incremental increases"
        elif 0 < roi < 1 and saturation == "high_saturation":
            recommendation = "Maintain/Reduce"
            action = "Low efficiency; approaching saturation"
        elif roi < 0:
            recommendation = "Reduce/Reallocate"
            action = "Negative contribution; reallocate to higher-performing channels"
        else:
            recommendation = "Maintain"
            action = "Keep current investment level"

        recommendations.append(
            {
                "channel": channel,
                "current_investment": investment,
                "roi": roi,
                "saturation_status": saturation,
                "recommendation": recommendation,
                "action": action,
            }
        )

    return pd.DataFrame(recommendations)


def run_full_insights_pipeline() -> dict:
    """Run complete insights extraction pipeline.

    Returns:
        Dictionary with all insights and visualizations
    """
    logger.info("=" * 60)
    logger.info("Running Insights Extraction Pipeline")
    logger.info("=" * 60)

    # Build models
    model_results = build_and_evaluate_models()
    df = model_results["df"]

    results = {"model_results": model_results}

    # Get average prices
    avg_offline_price = df["offline_price"].mean()
    avg_online_price = df["online_price"].mean()

    # ============ OFFLINE CHANNEL ============
    logger.info("\n" + "=" * 40)
    logger.info("OFFLINE CHANNEL INSIGHTS")
    logger.info("=" * 40)

    # Due-To attribution
    offline_due_to = calculate_due_to_sales(model_results, "offline")
    results["offline_due_to"] = offline_due_to
    logger.info("\nDue-To Attribution (Offline):")
    print(offline_due_to.to_string(index=False))

    # ROI calculation
    offline_roi = calculate_roi(offline_due_to, df, avg_offline_price)
    results["offline_roi"] = offline_roi
    logger.info("\nROI Analysis (Offline):")
    print(
        offline_roi[
            ["channel", "investment", "due_to_units", "incremental_revenue", "roi"]
        ].to_string(index=False)
    )

    # Bubble chart
    bubble_offline = create_bubble_chart(offline_roi, "Offline Channel")
    if bubble_offline:
        logger.success(f"Saved bubble chart: {bubble_offline}")

    # ============ ONLINE CHANNEL ============
    logger.info("\n" + "=" * 40)
    logger.info("ONLINE CHANNEL INSIGHTS")
    logger.info("=" * 40)

    # Due-To attribution
    online_due_to = calculate_due_to_sales(model_results, "online")
    results["online_due_to"] = online_due_to
    logger.info("\nDue-To Attribution (Online):")
    print(online_due_to.to_string(index=False))

    # ROI calculation
    online_roi = calculate_roi(online_due_to, df, avg_online_price)
    results["online_roi"] = online_roi
    logger.info("\nROI Analysis (Online):")
    print(
        online_roi[
            ["channel", "investment", "due_to_units", "incremental_revenue", "roi"]
        ].to_string(index=False)
    )

    # Bubble chart
    bubble_online = create_bubble_chart(online_roi, "Online Channel")
    if bubble_online:
        logger.success(f"Saved bubble chart: {bubble_online}")

    # ============ OPTIMIZATION RECOMMENDATIONS ============
    logger.info("\n" + "=" * 40)
    logger.info("OPTIMIZATION RECOMMENDATIONS")
    logger.info("=" * 40)

    # Combine both channels
    offline_rec = generate_optimization_recommendations(offline_roi, df)
    offline_rec["target_channel"] = "offline"
    online_rec = generate_optimization_recommendations(online_roi, df)
    online_rec["target_channel"] = "online"

    all_recommendations = pd.concat([offline_rec, online_rec], ignore_index=True)
    results["recommendations"] = all_recommendations

    logger.info("\nTop Recommendations:")
    priority_recs = all_recommendations[
        all_recommendations["recommendation"].isin(
            ["Significantly Increase", "Reduce/Reallocate"]
        )
    ]
    print(
        priority_recs[
            ["target_channel", "channel", "roi", "recommendation", "action"]
        ].to_string(index=False)
    )

    # Save all results
    offline_due_to.to_csv(OUTPUT_DIR / "offline_due_to.csv", index=False)
    online_due_to.to_csv(OUTPUT_DIR / "online_due_to.csv", index=False)
    offline_roi.to_csv(OUTPUT_DIR / "offline_roi.csv", index=False)
    online_roi.to_csv(OUTPUT_DIR / "online_roi.csv", index=False)
    all_recommendations.to_csv(OUTPUT_DIR / "recommendations.csv", index=False)

    logger.success(f"All insights saved to {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    results = run_full_insights_pipeline()
