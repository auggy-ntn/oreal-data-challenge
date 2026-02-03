"""Data loading and preprocessing module for L'Oréal MMM.

This module handles loading the Excel dataset and transforming it into
a unified weekly dataframe with all KPIs, Commercial Variables, and A&P data.
"""

import pandas as pd

from constants.paths import DATA_DIR
from src.utils.logger import logger

# File path
DATA_FILE = (
    DATA_DIR / "bronze" / "Dataset UK L'Oreal Paris Haircare - HEC Training.xlsx"
)

# Column name mappings for cleaner variable names
KPI_COLUMNS = {
    "Starting Week": "week",
    "Year": "year",
    "UK L'Oreal Paris Haircare Total Offline Sellout Value (in pound)": "offline_value",
    "UK L'Oreal Paris Haircare Total Offline Sellout Units": "offline_units",
    "UK L'Oreal Paris Haircare Total Online Sellout Value (in pound)": "online_value",
    "UK L'Oreal Paris Haircare Total Online Sellout Units": "online_units",
}

COMMERCIAL_COLUMNS = {
    "Starting Week": "week",
    "Year": "year",
    "UK L'Oreal Paris Haircare Offline Average Price (in pound)": "offline_price",
    "UK L'Oreal Paris Haircare Online Average Price (in pound)": "online_price",
    "UK L'Oreal Paris Haircare Total Weigheted Promotion Distribution (%)": (
        "promo_distribution"
    ),
}


def load_kpi_data() -> pd.DataFrame:
    """Load and rename KPI data."""
    logger.info("Loading KPI data...")
    df = pd.read_excel(DATA_FILE, sheet_name="KPI to model")
    df = df.rename(columns=KPI_COLUMNS)
    df["week"] = pd.to_datetime(df["week"])
    logger.success(f"Loaded {len(df)} weeks of KPI data")
    return df


def load_commercial_data() -> pd.DataFrame:
    """Load and rename Commercial Variables data."""
    logger.info("Loading Commercial data...")
    df = pd.read_excel(DATA_FILE, sheet_name="Commercial Variables")
    df = df.rename(columns=COMMERCIAL_COLUMNS)
    df["week"] = pd.to_datetime(df["week"])
    # Drop duplicate 'year' column - will use from KPI
    df = df.drop(columns=["year"])
    logger.success(f"Loaded {len(df)} weeks of Commercial data")
    return df


def load_ap_data() -> pd.DataFrame:
    """Load A&P Variables data."""
    logger.info("Loading A&P data...")
    df = pd.read_excel(DATA_FILE, sheet_name="A&P Variables")
    df = df.rename(columns={"Starting week": "week"})
    df["week"] = pd.to_datetime(df["week"])
    logger.success(
        f"Loaded {len(df)} A&P records ({df['growth_driver_l5'].nunique()} touchpoints)"
    )
    return df


def pivot_ap_data(ap_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot A&P data to wide format with one column per touchpoint.

    Creates columns for both investment and execution metrics:
    - {touchpoint}_investment: Spend in pounds
    - {touchpoint}_execution: Impressions/Engagements/GRPs/Units

    Args:
        ap_df: Raw A&P dataframe with hierarchical structure

    Returns:
        Wide-format dataframe with week as index and touchpoint columns
    """
    logger.info("Pivoting A&P data to wide format...")

    # Create unique touchpoint identifier
    ap_df = ap_df.copy()
    ap_df["touchpoint"] = ap_df["growth_driver_l5"].str.lower().str.replace(" ", "_")

    # Pivot investment
    investment_pivot = ap_df.pivot_table(
        index="week",
        columns="touchpoint",
        values="investment (in pound)",
        aggfunc="sum",
        fill_value=0,
    )
    investment_pivot.columns = [f"{col}_investment" for col in investment_pivot.columns]

    # Pivot execution
    execution_pivot = ap_df.pivot_table(
        index="week",
        columns="touchpoint",
        values="execution",
        aggfunc="sum",
        fill_value=0,
    )
    execution_pivot.columns = [f"{col}_execution" for col in execution_pivot.columns]

    # Combine both pivots
    ap_wide = pd.concat([investment_pivot, execution_pivot], axis=1)
    ap_wide = ap_wide.reset_index()

    logger.success(f"Created {len(ap_wide.columns) - 1} A&P feature columns")
    return ap_wide


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for seasonality and trend.

    Args:
        df: Dataframe with 'week' column

    Returns:
        DataFrame with added time features
    """
    df = df.copy()

    # Week of year for seasonality
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)

    # Month for broader seasonality
    df["month"] = df["week"].dt.month

    # Quarter
    df["quarter"] = df["week"].dt.quarter

    # Week number (trend proxy)
    df["week_num"] = range(len(df))

    # Year indicator
    df["year_2023"] = (df["year"] == 2023).astype(int)

    # Holiday flags (UK major retail periods)
    df["is_christmas"] = ((df["month"] == 12) & (df["week_of_year"] >= 49)).astype(int)
    df["is_easter"] = (
        (df["month"] == 4) & (df["week_of_year"].isin([14, 15, 16]))
    ).astype(int)
    df["is_black_friday"] = (
        (df["month"] == 11) & (df["week_of_year"].isin([47, 48]))
    ).astype(int)

    logger.info("Added time-based features")
    return df


def load_unified_dataset() -> pd.DataFrame:
    """Load and merge all data sources into a unified weekly dataset.

    Returns:
        Unified DataFrame with:
        - KPIs (offline/online sales)
        - Commercial variables (price, promotion)
        - A&P variables (investment & execution per touchpoint)
        - Time features (seasonality, trend, holidays)
    """
    logger.info("Building unified dataset...")

    # Load individual datasets
    kpi_df = load_kpi_data()
    commercial_df = load_commercial_data()
    ap_df = load_ap_data()

    # Pivot A&P to wide format
    ap_wide = pivot_ap_data(ap_df)

    # Merge all datasets
    df = kpi_df.merge(commercial_df, on="week", how="left")
    df = df.merge(ap_wide, on="week", how="left")

    # Add time features
    df = create_time_features(df)

    # Sort by week
    df = df.sort_values("week").reset_index(drop=True)

    # Fill any NaN values in A&P columns with 0
    ap_cols = [col for col in df.columns if "_investment" in col or "_execution" in col]
    df[ap_cols] = df[ap_cols].fillna(0)

    logger.success(
        f"Unified dataset created: {df.shape[0]} rows, {df.shape[1]} columns"
    )

    return df


def get_touchpoint_names() -> list[str]:
    """Get list of unique touchpoint names."""
    return [
        "amazon",
        "amazon_retail",
        "bvod",
        "citrus",
        "criteo",
        "google",
        "google_video",
        "influencer_management",
        "linear",
        "meta",
        "meta_collab_ads",
        "pinterest",
        "tesco",
        "testers_and_merchandising",
        "the_hut_group",
        "tik_tok",
        "youtube",
    ]


if __name__ == "__main__":
    # Test data loading
    df = load_unified_dataset()
    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nDate range: {df['week'].min()} to {df['week'].max()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
