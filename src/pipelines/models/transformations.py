"""BETiq-compliant transformations for Marketing Mix Modeling.

This module implements the standard MMM transformations:
- Adstock (Decay): Captures the memory effect of advertising
- Saturation (Hill Function): Models diminishing returns
- Lag: Accounts for delayed response to advertising

These transformations are applied to A&P execution data before regression.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.logger import logger


def adstock(x: np.ndarray, decay_rate: float) -> np.ndarray:
    """Apply geometric adstock decay transformation.

    The adstock effect captures how advertising impact carries over time.
    Formula: A_t = X_t + λ * A_{t-1}

    Args:
        x: Array of weekly execution values
        decay_rate: Decay rate λ (0-1). Higher = longer memory.
                   Typical values: TV 0.8-0.9, Digital 0.4-0.6

    Returns:
        Adstocked array of same length as input
    """
    if decay_rate < 0 or decay_rate > 1:
        raise ValueError("decay_rate must be between 0 and 1")

    adstocked = np.zeros(len(x))
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay_rate * adstocked[i - 1]
    return adstocked


def hill_saturation(x: np.ndarray, K: float, S: float) -> np.ndarray:
    """Apply Hill function saturation transformation.

    Models diminishing returns as spend increases.
    Formula: y = x^S / (K^S + x^S)

    Args:
        x: Array of execution values (should be normalized)
        K: Half-saturation point (50% effect achieved at this level)
        S: Shape parameter (steepness). Higher = sharper saturation.
           S=1 is standard Michaelis-Menten, S>1 is S-curve

    Returns:
        Saturated array with values between 0 and 1
    """
    if K <= 0:
        raise ValueError("K must be positive")
    if S <= 0:
        raise ValueError("S must be positive")

    # Handle zero values
    x = np.maximum(x, 0)

    return np.power(x, S) / (np.power(K, S) + np.power(x, S))


def lag_transform(x: np.ndarray, lag_weeks: int) -> np.ndarray:
    """Apply lag transformation to account for delayed response.

    Args:
        x: Array of execution values
        lag_weeks: Number of weeks to shift (1-4 typical)

    Returns:
        Lagged array (first lag_weeks values will be 0)
    """
    if lag_weeks < 0:
        raise ValueError("lag_weeks must be non-negative")
    if lag_weeks == 0:
        return x

    lagged = np.zeros(len(x))
    lagged[lag_weeks:] = x[:-lag_weeks]
    return lagged


def normalize_for_saturation(x: np.ndarray) -> np.ndarray:
    """Normalize values for saturation function.

    Scales to 0-1 range based on max value.

    Args:
        x: Raw execution values

    Returns:
        Normalized array
    """
    max_val = np.max(x)
    if max_val == 0:
        return x
    return x / max_val


def apply_full_transformation(
    x: np.ndarray,
    adstock_rate: float = 0.5,
    saturation_k: float = 0.5,
    saturation_s: float = 1.5,
    lag_weeks: int = 0,
) -> np.ndarray:
    """Apply full transformation pipeline: Lag -> Adstock -> Saturation.

    Args:
        x: Raw execution values
        adstock_rate: Decay rate for adstock (0-1)
        saturation_k: Half-saturation point (0-1, relative to max)
        saturation_s: Saturation shape parameter
        lag_weeks: Lag in weeks

    Returns:
        Fully transformed array
    """
    # Step 1: Apply lag
    transformed = lag_transform(x, lag_weeks)

    # Step 2: Apply adstock
    transformed = adstock(transformed, adstock_rate)

    # Step 3: Normalize and apply saturation
    max_val = np.max(transformed)
    if max_val > 0:
        K = saturation_k * max_val  # K relative to max adstocked value
        transformed = hill_saturation(transformed, K, saturation_s)

    return transformed


# Default parameters by channel type (based on industry benchmarks)
DEFAULT_PARAMS = {
    # TV has high carryover, moderate saturation
    "linear": {"adstock": 0.85, "saturation_k": 0.5, "saturation_s": 2.0, "lag": 0},
    "bvod": {"adstock": 0.70, "saturation_k": 0.5, "saturation_s": 1.8, "lag": 0},
    # Digital video
    "youtube": {"adstock": 0.60, "saturation_k": 0.5, "saturation_s": 1.5, "lag": 0},
    "google_video": {
        "adstock": 0.60,
        "saturation_k": 0.5,
        "saturation_s": 1.5,
        "lag": 0,
    },
    # Social media - faster decay
    "meta": {"adstock": 0.50, "saturation_k": 0.5, "saturation_s": 1.5, "lag": 0},
    "tik_tok": {"adstock": 0.40, "saturation_k": 0.5, "saturation_s": 1.5, "lag": 0},
    "pinterest": {"adstock": 0.45, "saturation_k": 0.5, "saturation_s": 1.5, "lag": 0},
    # Search - very fast decay (intent-driven)
    "google": {"adstock": 0.30, "saturation_k": 0.5, "saturation_s": 1.2, "lag": 0},
    "amazon": {"adstock": 0.25, "saturation_k": 0.5, "saturation_s": 1.2, "lag": 0},
    "citrus": {"adstock": 0.25, "saturation_k": 0.5, "saturation_s": 1.2, "lag": 0},
    "criteo": {"adstock": 0.25, "saturation_k": 0.5, "saturation_s": 1.2, "lag": 0},
    # Retail media
    "amazon_retail": {
        "adstock": 0.35,
        "saturation_k": 0.5,
        "saturation_s": 1.3,
        "lag": 0,
    },
    "tesco": {"adstock": 0.35, "saturation_k": 0.5, "saturation_s": 1.3, "lag": 0},
    "the_hut_group": {
        "adstock": 0.35,
        "saturation_k": 0.5,
        "saturation_s": 1.3,
        "lag": 0,
    },
    # Influencer - longer effect
    "influencer_management": {
        "adstock": 0.65,
        "saturation_k": 0.5,
        "saturation_s": 1.5,
        "lag": 1,
    },
    # Collaboration ads
    "meta_collab_ads": {
        "adstock": 0.50,
        "saturation_k": 0.5,
        "saturation_s": 1.5,
        "lag": 0,
    },
    # In-store
    "testers_and_merchandising": {
        "adstock": 0.80,
        "saturation_k": 0.5,
        "saturation_s": 2.0,
        "lag": 0,
    },
}


def get_default_params(channel: str) -> dict:
    """Get default transformation parameters for a channel.

    Args:
        channel: Channel name (e.g., 'meta', 'linear')

    Returns:
        Dictionary with adstock, saturation_k, saturation_s, lag parameters
    """
    return DEFAULT_PARAMS.get(
        channel,
        {"adstock": 0.50, "saturation_k": 0.5, "saturation_s": 1.5, "lag": 0},
    )


def transform_all_channels(
    df: pd.DataFrame,
    params: dict | None = None,
    use_defaults: bool = True,
) -> pd.DataFrame:
    """Apply transformations to all A&P execution columns.

    Args:
        df: DataFrame with execution columns (e.g., 'meta_execution')
        params: Optional dict of {channel: {adstock, saturation_k, saturation_s, lag}}
        use_defaults: If True, use default params for channels not in params

    Returns:
        DataFrame with new transformed columns (e.g., 'meta_transformed')
    """
    logger.info("Applying transformations to all A&P channels...")

    df = df.copy()
    execution_cols = [col for col in df.columns if col.endswith("_execution")]

    if params is None:
        params = {}

    for col in execution_cols:
        channel = col.replace("_execution", "")

        # Get parameters
        if channel in params:
            p = params[channel]
        elif use_defaults:
            p = get_default_params(channel)
        else:
            continue

        # Apply transformation
        transformed = apply_full_transformation(
            df[col].values,
            adstock_rate=p["adstock"],
            saturation_k=p["saturation_k"],
            saturation_s=p["saturation_s"],
            lag_weeks=p["lag"],
        )

        df[f"{channel}_transformed"] = transformed
        logger.debug(f"Transformed {channel}: adstock={p['adstock']:.2f}")

    transformed_cols = [col for col in df.columns if col.endswith("_transformed")]
    logger.success(f"Created {len(transformed_cols)} transformed features")

    return df


def optimize_adstock(
    x: np.ndarray,
    y: np.ndarray,
    saturation_k: float = 0.5,
    saturation_s: float = 1.5,
) -> tuple[float, float]:
    """Optimize adstock decay rate to maximize correlation with target.

    Args:
        x: Execution values
        y: Target variable (sales)
        saturation_k: Fixed saturation K parameter
        saturation_s: Fixed saturation S parameter

    Returns:
        Tuple of (optimal_decay_rate, correlation)
    """

    def neg_correlation(decay_rate):
        transformed = apply_full_transformation(
            x,
            adstock_rate=decay_rate[0],
            saturation_k=saturation_k,
            saturation_s=saturation_s,
        )
        # Handle zero variance
        if np.std(transformed) == 0:
            return 0
        corr = np.corrcoef(transformed, y)[0, 1]
        return -corr if not np.isnan(corr) else 0

    result = minimize(
        neg_correlation,
        x0=[0.5],
        bounds=[(0.05, 0.95)],
        method="L-BFGS-B",
    )

    optimal_decay = result.x[0]
    correlation = -result.fun

    return optimal_decay, correlation


def get_saturation_curve_points(K: float, S: float) -> dict:
    """Calculate ABCD points on the saturation curve.

    A: Threshold (5% of max effect)
    B: Linear growth (25% of max effect)
    C: Saturation begins (75% of max effect)
    D: Full saturation (95% of max effect)

    Args:
        K: Half-saturation parameter
        S: Shape parameter

    Returns:
        Dictionary with A, B, C, D points as (x, y) tuples
    """
    # For Hill function: y = x^S / (K^S + x^S)
    # Solving for x given y: x = K * (y / (1 - y))^(1/S)

    def inverse_hill(y):
        if y >= 1 or y <= 0:
            return np.nan
        return K * np.power(y / (1 - y), 1 / S)

    points = {
        "A": (inverse_hill(0.05), 0.05),
        "B": (inverse_hill(0.25), 0.25),
        "C": (inverse_hill(0.75), 0.75),
        "D": (inverse_hill(0.95), 0.95),
    }

    return points


if __name__ == "__main__":
    # Demo transformations
    from src.pipelines.data.data_loader import load_unified_dataset

    df = load_unified_dataset()

    # Apply transformations
    df_transformed = transform_all_channels(df)

    # Show sample
    transformed_cols = [col for col in df_transformed.columns if "_transformed" in col]
    print(f"\nTransformed columns: {transformed_cols}")
    print("\nSample data:")
    print(df_transformed[["week", "meta_execution", "meta_transformed"]].head(10))

    # Optimize adstock for meta
    optimal_decay, corr = optimize_adstock(
        df["meta_execution"].values,
        df["offline_units"].values,
    )
    print(
        f"\nOptimal adstock for Meta vs Offline Units: "
        f"{optimal_decay:.3f} (corr={corr:.3f})"
    )

    # Show saturation points
    points = get_saturation_curve_points(K=0.5, S=1.5)
    print("\nSaturation curve ABCD points (K=0.5, S=1.5):")
    for name, (x, y) in points.items():
        print(f"  {name}: x={x:.3f}, y={y:.1%}")
