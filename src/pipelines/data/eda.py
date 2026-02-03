"""Exploratory Data Analysis module for L'Oréal MMM.

This module provides comprehensive EDA including:
- Time-series decomposition
- Correlation analysis
- VIF multicollinearity check
- Visual spike analysis
- Stationarity testing (ADF)
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from constants.paths import PROJECT_ROOT
from src.pipelines.data.data_loader import load_unified_dataset
from src.utils.logger import logger

warnings.filterwarnings("ignore")

# Output directory for plots
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def time_series_decomposition(
    df: pd.DataFrame, target_col: str, period: int = 52
) -> dict:
    """Decompose time series into trend, seasonality, and residuals.

    Args:
        df: DataFrame with weekly data
        target_col: Column to decompose ('offline_units' or 'online_units')
        period: Seasonality period (52 for weekly data with yearly seasonality)

    Returns:
        Dictionary with decomposition components and plot path
    """
    logger.info(f"Decomposing {target_col}...")

    series = df[target_col].values

    # STL decomposition (robust to outliers)
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(
        f"Time Series Decomposition: {target_col}", fontsize=14, fontweight="bold"
    )

    axes[0].plot(df["week"], series, color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Original")
    axes[0].set_title("Original Series")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["week"], result.trend, color="darkorange", linewidth=1.5)
    axes[1].set_ylabel("Trend")
    axes[1].set_title("Trend Component")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["week"], result.seasonal, color="green", linewidth=1.5)
    axes[2].set_ylabel("Seasonal")
    axes[2].set_title("Seasonal Component")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["week"], result.resid, color="red", linewidth=1.5)
    axes[3].set_ylabel("Residual")
    axes[3].set_title("Residual Component")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Week")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"decomposition_{target_col}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(f"Saved decomposition plot to {plot_path}")

    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "plot_path": str(plot_path),
    }


def correlation_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    """Generate correlation matrix between A&P execution and sales.

    Args:
        df: Unified dataset

    Returns:
        Tuple of (correlation DataFrame, plot path)
    """
    logger.info("Computing correlation matrix...")

    # Select execution columns and target columns
    execution_cols = [col for col in df.columns if "_execution" in col]
    target_cols = ["offline_units", "online_units"]
    analysis_cols = target_cols + execution_cols

    # Compute correlation
    corr_matrix = df[analysis_cols].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 12))

    # Focus on correlations with targets
    target_corr = corr_matrix[target_cols].drop(target_cols)

    sns.heatmap(
        target_corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_title("Correlation: A&P Execution vs. Sales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Variables")
    ax.set_ylabel("A&P Touchpoints (Execution)")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(f"Saved correlation heatmap to {plot_path}")

    return target_corr, plot_path


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factor for multicollinearity check.

    Args:
        df: Unified dataset

    Returns:
        DataFrame with VIF values for each predictor
    """
    logger.info("Calculating VIF for multicollinearity check...")

    # Select numeric predictors (excluding targets and identifiers)
    exclude_cols = [
        "week",
        "year",
        "offline_units",
        "online_units",
        "offline_value",
        "online_value",
    ]
    predictor_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and df[col].dtype in ["float64", "int64", "int32"]
    ]

    # Remove columns with zero variance
    non_zero_var_cols = [col for col in predictor_cols if df[col].std() > 0]

    # Create design matrix
    X = df[non_zero_var_cols].copy()
    X = X.fillna(0)

    # Calculate VIF (limit to avoid computational issues)
    # Focus on execution columns for A&P
    execution_cols = [col for col in non_zero_var_cols if "_execution" in col]
    control_cols = ["offline_price", "online_price", "promo_distribution", "week_num"]
    control_cols = [c for c in control_cols if c in non_zero_var_cols]

    vif_cols = control_cols + execution_cols
    X_vif = X[vif_cols]

    vif_data = []
    for i, col in enumerate(X_vif.columns):
        try:
            vif = variance_inflation_factor(X_vif.values, i)
            vif_data.append({"variable": col, "VIF": vif})
        except Exception:
            vif_data.append({"variable": col, "VIF": np.nan})

    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    # Flag high VIF
    vif_df["multicollinearity_risk"] = vif_df["VIF"].apply(
        lambda x: "HIGH" if x > 10 else ("MODERATE" if x > 5 else "LOW")
    )

    logger.success(f"VIF calculated for {len(vif_df)} variables")

    return vif_df


def visual_spike_analysis(df: pd.DataFrame) -> Path:
    """Plot sales against major media bursts for visual due-to analysis.

    Args:
        df: Unified dataset

    Returns:
        Path to saved plot
    """
    logger.info("Creating visual spike analysis...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Sales (both channels)
    ax1 = axes[0]
    ax1.plot(
        df["week"],
        df["offline_units"],
        label="Offline Units",
        color="steelblue",
        linewidth=1.5,
    )
    ax1.plot(
        df["week"],
        df["online_units"],
        label="Online Units",
        color="coral",
        linewidth=1.5,
    )
    ax1.set_ylabel("Units Sold")
    ax1.set_title("Sales Over Time", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Highlight holiday periods
    for idx, row in df.iterrows():
        if row["is_christmas"] == 1:
            ax1.axvspan(
                row["week"],
                row["week"] + pd.Timedelta(days=7),
                alpha=0.2,
                color="red",
                label="_Christmas" if idx == 0 else "",
            )
        if row["is_black_friday"] == 1:
            ax1.axvspan(
                row["week"],
                row["week"] + pd.Timedelta(days=7),
                alpha=0.2,
                color="purple",
                label="_Black Friday" if idx == 0 else "",
            )

    # Plot 2: TV Spend (Linear + BVOD)
    ax2 = axes[1]
    ax2.fill_between(
        df["week"],
        df["linear_execution"],
        alpha=0.7,
        label="Linear TV (GRPs)",
        color="darkgreen",
    )
    ax2.fill_between(
        df["week"],
        df["bvod_execution"] / 1e6,
        alpha=0.7,
        label="BVOD (Impressions, M)",
        color="lightgreen",
    )
    ax2.set_ylabel("TV Execution")
    ax2.set_title("TV Media (Linear + BVOD)", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Social Media
    ax3 = axes[2]
    ax3.fill_between(
        df["week"],
        df["meta_execution"] / 1e6,
        alpha=0.7,
        label="Meta (M)",
        color="blue",
    )
    ax3.fill_between(
        df["week"],
        df["tik_tok_execution"] / 1e6,
        alpha=0.7,
        label="TikTok (M)",
        color="black",
    )
    ax3.fill_between(
        df["week"],
        df["youtube_execution"] / 1e6,
        alpha=0.7,
        label="YouTube (M)",
        color="red",
    )
    ax3.set_ylabel("Social Execution (M)")
    ax3.set_title("Social Media Execution", fontweight="bold")
    ax3.set_xlabel("Week")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "spike_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(f"Saved spike analysis to {plot_path}")

    return plot_path


def adf_stationarity_test(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Run Augmented Dickey-Fuller test for stationarity.

    Args:
        df: Unified dataset
        columns: Columns to test

    Returns:
        DataFrame with ADF test results
    """
    logger.info("Running ADF stationarity tests...")

    results = []
    for col in columns:
        series = df[col].dropna()
        try:
            adf_result = adfuller(series, autolag="AIC")
            results.append(
                {
                    "variable": col,
                    "adf_statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "lags_used": adf_result[2],
                    "observations": adf_result[3],
                    "critical_1%": adf_result[4]["1%"],
                    "critical_5%": adf_result[4]["5%"],
                    "critical_10%": adf_result[4]["10%"],
                    "is_stationary": adf_result[1] < 0.05,
                }
            )
        except Exception as e:
            logger.warning(f"ADF test failed for {col}: {e}")

    adf_df = pd.DataFrame(results)
    logger.success(f"ADF tests completed for {len(adf_df)} variables")

    return adf_df


def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for key variables.

    Args:
        df: Unified dataset

    Returns:
        Summary statistics DataFrame
    """
    # Key columns for summary
    key_cols = [
        "offline_units",
        "online_units",
        "offline_value",
        "online_value",
        "offline_price",
        "online_price",
        "promo_distribution",
    ]

    # Add top A&P channels by total spend
    investment_cols = [col for col in df.columns if "_investment" in col]
    top_channels = df[investment_cols].sum().nlargest(5).index.tolist()
    execution_equiv = [col.replace("_investment", "_execution") for col in top_channels]

    summary_cols = key_cols + execution_equiv

    summary = df[summary_cols].describe().T
    summary["total"] = df[summary_cols].sum()

    return summary


def run_full_eda(save_outputs: bool = True) -> dict:
    """Run complete EDA pipeline.

    Args:
        save_outputs: Whether to save plots and tables

    Returns:
        Dictionary containing all EDA results
    """
    logger.info("=" * 50)
    logger.info("Starting Full EDA Pipeline")
    logger.info("=" * 50)

    # Load data
    df = load_unified_dataset()

    results = {}

    # 1. Summary statistics
    logger.info("\n--- Summary Statistics ---")
    results["summary_stats"] = generate_summary_stats(df)
    print(results["summary_stats"])

    # 2. Time series decomposition
    logger.info("\n--- Time Series Decomposition ---")
    results["decomp_offline"] = time_series_decomposition(df, "offline_units")
    results["decomp_online"] = time_series_decomposition(df, "online_units")

    # 3. Correlation analysis
    logger.info("\n--- Correlation Analysis ---")
    corr_df, corr_plot = correlation_analysis(df)
    results["correlation"] = corr_df
    results["correlation_plot"] = corr_plot
    print("\nTop correlations with Offline Units:")
    print(corr_df["offline_units"].sort_values(ascending=False).head(10))
    print("\nTop correlations with Online Units:")
    print(corr_df["online_units"].sort_values(ascending=False).head(10))

    # 4. VIF analysis
    logger.info("\n--- VIF Multicollinearity Check ---")
    vif_df = calculate_vif(df)
    results["vif"] = vif_df
    print("\nHigh VIF variables (>10):")
    print(vif_df[vif_df["VIF"] > 10])

    # 5. Visual spike analysis
    logger.info("\n--- Visual Spike Analysis ---")
    spike_plot = visual_spike_analysis(df)
    results["spike_plot"] = spike_plot

    # 6. Stationarity tests
    logger.info("\n--- Stationarity Tests (ADF) ---")
    test_cols = ["offline_units", "online_units", "offline_price", "promo_distribution"]
    adf_df = adf_stationarity_test(df, test_cols)
    results["adf_tests"] = adf_df
    print("\nADF Test Results:")
    print(adf_df[["variable", "adf_statistic", "p_value", "is_stationary"]])

    # Save VIF and ADF as CSV
    if save_outputs:
        vif_df.to_csv(OUTPUT_DIR / "vif_analysis.csv", index=False)
        adf_df.to_csv(OUTPUT_DIR / "adf_tests.csv", index=False)
        corr_df.to_csv(OUTPUT_DIR / "correlations.csv")
        logger.success(f"All EDA outputs saved to {OUTPUT_DIR}")

    logger.info("=" * 50)
    logger.info("EDA Pipeline Complete!")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    results = run_full_eda()
