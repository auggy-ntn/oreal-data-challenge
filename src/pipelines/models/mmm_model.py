"""Marketing Mix Model implementation for L'Oréal Haircare.

This module builds and validates two separate OLS regression models:
1. Offline Units Model: Explains physical store sales
2. Online Units Model: Explains e-commerce sales

The models incorporate:
- Control variables (price, promotion, seasonality, trend)
- Transformed A&P execution features (adstock + saturation applied)
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from constants.paths import PROJECT_ROOT
from src.pipelines.data.data_loader import load_unified_dataset
from src.pipelines.models.transformations import transform_all_channels
from src.utils.logger import logger

warnings.filterwarnings("ignore")

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_model_data(
    df: pd.DataFrame,
    target: str,
    include_transformed: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target for regression.

    Args:
        df: Unified dataset with transformations applied
        target: Target column name ('offline_units' or 'online_units')
        include_transformed: Whether to include transformed A&P features

    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    # Control variables
    control_cols = [
        "week_num",  # Trend
        "promo_distribution",  # Promotion
    ]

    # Add appropriate price column
    if target == "offline_units":
        control_cols.append("offline_price")
    else:
        control_cols.append("online_price")

    # Seasonality - use Fourier terms for smoother representation
    df = df.copy()
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_week_2"] = np.sin(4 * np.pi * df["week_of_year"] / 52)
    df["cos_week_2"] = np.cos(4 * np.pi * df["week_of_year"] / 52)

    seasonality_cols = ["sin_week", "cos_week", "sin_week_2", "cos_week_2"]

    # Holiday dummies
    holiday_cols = ["is_christmas", "is_black_friday"]

    # A&P transformed features
    if include_transformed:
        ap_cols = [col for col in df.columns if col.endswith("_transformed")]
    else:
        ap_cols = []

    # Combine all features
    all_feature_cols = control_cols + seasonality_cols + holiday_cols + ap_cols

    # Filter to columns that exist
    feature_cols = [col for col in all_feature_cols if col in df.columns]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Handle any NaN
    X = X.fillna(0)

    return X, y


def calculate_model_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for model features.

    Args:
        X: Feature matrix

    Returns:
        DataFrame with VIF values
    """
    # Add constant for VIF calculation
    X_const = sm.add_constant(X)

    vif_data = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X_const.values, i)
            vif_data.append({"variable": col, "VIF": vif})
        except Exception:
            vif_data.append({"variable": col, "VIF": np.nan})

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def fit_ols_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS regression model and print diagnostics.

    Args:
        X: Feature matrix
        y: Target variable
        model_name: Name for logging

    Returns:
        Fitted OLS model
    """
    logger.info(f"Fitting {model_name}...")

    # Add constant (intercept)
    X_const = sm.add_constant(X)

    # Fit OLS
    model = sm.OLS(y, X_const).fit()

    # Print summary stats
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"R-squared: {model.rsquared:.4f}")
    logger.info(f"Adj. R-squared: {model.rsquared_adj:.4f}")
    logger.info(f"F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.4e})")
    logger.info(f"Durbin-Watson: {durbin_watson(model.resid):.4f}")
    logger.info(f"Observations: {model.nobs:.0f}")
    logger.info(f"Parameters: {model.df_model + 1:.0f}")
    logger.info(f"Obs/Param ratio: {model.nobs / (model.df_model + 1):.1f}")

    return model


def stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    significance_level: float = 0.10,
) -> list[str]:
    """Forward stepwise selection based on p-values.

    Args:
        X: Feature matrix
        y: Target
        significance_level: Maximum p-value to include variable

    Returns:
        List of selected feature names
    """
    logger.info("Running stepwise feature selection...")

    remaining = list(X.columns)
    selected = []

    while remaining:
        best_pval = 1.0
        best_feature = None

        for feature in remaining:
            candidate = selected + [feature]
            X_candidate = sm.add_constant(X[candidate])
            model = sm.OLS(y, X_candidate).fit()
            pval = model.pvalues[feature]

            if pval < best_pval:
                best_pval = pval
                best_feature = feature

        if best_pval < significance_level:
            selected.append(best_feature)
            remaining.remove(best_feature)
            logger.debug(f"Added {best_feature} (p={best_pval:.4f})")
        else:
            break

    logger.info(f"Selected {len(selected)} features")
    return selected


def create_model_diagnostics_plot(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    y: pd.Series,
    model_name: str,
) -> Path:
    """Create diagnostic plots for regression model.

    Args:
        model: Fitted OLS model
        y: Actual target values
        model_name: Name for plot title

    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Model Diagnostics: {model_name}", fontsize=14, fontweight="bold")

    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    y_pred = model.fittedvalues
    ax1.scatter(y, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"Actual vs Predicted (R²={model.rsquared:.3f})")
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs Fitted
    ax2 = axes[0, 1]
    residuals = model.resid
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax2.axhline(y=0, color="r", linestyle="--")
    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Fitted")
    ax2.grid(True, alpha=0.3)

    # 3. Residuals distribution
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Residual Distribution")
    ax3.grid(True, alpha=0.3)

    # 4. Q-Q plot
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title("Q-Q Plot")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"diagnostics_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def create_coefficient_plot(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    model_name: str,
) -> Path:
    """Create coefficient bar plot.

    Args:
        model: Fitted OLS model
        model_name: Name for plot title

    Returns:
        Path to saved plot
    """
    # Extract coefficients (exclude constant)
    coefs = model.params.drop("const", errors="ignore")
    pvals = model.pvalues.drop("const", errors="ignore")

    # Sort by absolute value
    sorted_idx = coefs.abs().sort_values(ascending=True).index
    coefs = coefs[sorted_idx]
    pvals = pvals[sorted_idx]

    # Color by significance
    colors = ["green" if p < 0.05 else "orange" if p < 0.10 else "gray" for p in pvals]

    fig, ax = plt.subplots(figsize=(12, max(8, len(coefs) * 0.3)))

    y_pos = range(len(coefs))
    ax.barh(y_pos, coefs, color=colors, edgecolor="black", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs.index)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(
        f"Coefficients: {model_name}\n(Green: p<0.05, Orange: p<0.10, Gray: n.s.)"
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"coefficients_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def extract_ap_coefficients(
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    """Extract A&P channel coefficients from model.

    Args:
        model: Fitted OLS model

    Returns:
        DataFrame with channel, coefficient, p-value, significance
    """
    ap_params = [
        (name, coef, model.pvalues[name])
        for name, coef in model.params.items()
        if "_transformed" in name
    ]

    if not ap_params:
        return pd.DataFrame()

    df = pd.DataFrame(ap_params, columns=["channel", "coefficient", "p_value"])
    df["channel"] = df["channel"].str.replace("_transformed", "")
    df["significant"] = df["p_value"] < 0.05
    df = df.sort_values("coefficient", ascending=False).reset_index(drop=True)

    return df


def build_and_evaluate_models(
    use_stepwise: bool = False,
) -> dict:
    """Build and evaluate both Online and Offline models.

    Args:
        use_stepwise: Whether to use stepwise selection

    Returns:
        Dictionary with models and diagnostics
    """
    logger.info("=" * 60)
    logger.info("Building MMM Models")
    logger.info("=" * 60)

    # Load and transform data
    df = load_unified_dataset()
    df = transform_all_channels(df)

    results = {"df": df}

    # Build Offline Model
    logger.info("\n" + "=" * 40)
    logger.info("OFFLINE UNITS MODEL")
    logger.info("=" * 40)

    X_offline, y_offline = prepare_model_data(df, "offline_units")

    if use_stepwise:
        selected_offline = stepwise_selection(X_offline, y_offline)
        X_offline = X_offline[selected_offline]

    offline_model = fit_ols_model(X_offline, y_offline, "Offline Units")
    results["offline_model"] = offline_model
    results["offline_X"] = X_offline
    results["offline_y"] = y_offline

    # VIF check
    vif_offline = calculate_model_vif(X_offline)
    logger.info(
        f"\nVIF check - High VIF (>10): {(vif_offline['VIF'] > 10).sum()} variables"
    )

    # Create plots
    diag_path_off = create_model_diagnostics_plot(
        offline_model, y_offline, "Offline Units"
    )
    create_coefficient_plot(offline_model, "Offline Units")
    logger.success(f"Saved diagnostics: {diag_path_off}")

    # Build Online Model
    logger.info("\n" + "=" * 40)
    logger.info("ONLINE UNITS MODEL")
    logger.info("=" * 40)

    X_online, y_online = prepare_model_data(df, "online_units")

    if use_stepwise:
        selected_online = stepwise_selection(X_online, y_online)
        X_online = X_online[selected_online]

    online_model = fit_ols_model(X_online, y_online, "Online Units")
    results["online_model"] = online_model
    results["online_X"] = X_online
    results["online_y"] = y_online

    # VIF check
    vif_online = calculate_model_vif(X_online)
    logger.info(
        f"\nVIF check - High VIF (>10): {(vif_online['VIF'] > 10).sum()} variables"
    )

    # Create plots
    diag_path_on = create_model_diagnostics_plot(online_model, y_online, "Online Units")
    create_coefficient_plot(online_model, "Online Units")
    logger.success(f"Saved diagnostics: {diag_path_on}")

    # Extract A&P coefficients
    logger.info("\n" + "=" * 40)
    logger.info("A&P CHANNEL COEFFICIENTS")
    logger.info("=" * 40)

    offline_coefs = extract_ap_coefficients(offline_model)
    online_coefs = extract_ap_coefficients(online_model)

    results["offline_ap_coefs"] = offline_coefs
    results["online_ap_coefs"] = online_coefs

    logger.info("\nOffline - Significant A&P Channels:")
    sig_offline = offline_coefs[offline_coefs["significant"]]
    if len(sig_offline) > 0:
        print(sig_offline.to_string(index=False))
    else:
        logger.warning("No significant A&P channels at p<0.05")

    logger.info("\nOnline - Significant A&P Channels:")
    sig_online = online_coefs[online_coefs["significant"]]
    if len(sig_online) > 0:
        print(sig_online.to_string(index=False))
    else:
        logger.warning("No significant A&P channels at p<0.05")

    # Save summary
    summary = {
        "offline_r2": offline_model.rsquared,
        "offline_adj_r2": offline_model.rsquared_adj,
        "offline_dw": durbin_watson(offline_model.resid),
        "online_r2": online_model.rsquared,
        "online_adj_r2": online_model.rsquared_adj,
        "online_dw": durbin_watson(online_model.resid),
    }
    results["summary"] = summary

    # Save model summaries to file
    with open(OUTPUT_DIR / "offline_model_summary.txt", "w") as f:
        f.write(offline_model.summary().as_text())
    with open(OUTPUT_DIR / "online_model_summary.txt", "w") as f:
        f.write(online_model.summary().as_text())

    logger.info("\n" + "=" * 60)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Offline Model: R²={summary['offline_r2']:.4f}, DW={summary['offline_dw']:.2f}"
    )
    logger.info(
        f"Online Model:  R²={summary['online_r2']:.4f}, DW={summary['online_dw']:.2f}"
    )

    return results


if __name__ == "__main__":
    results = build_and_evaluate_models(use_stepwise=False)

    print("\n=== OFFLINE MODEL SUMMARY ===")
    print(results["offline_model"].summary().tables[0])

    print("\n=== ONLINE MODEL SUMMARY ===")
    print(results["online_model"].summary().tables[0])
