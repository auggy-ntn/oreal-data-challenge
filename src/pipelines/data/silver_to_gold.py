"""Silver to Gold pipeline."""

from pathlib import Path

import pandas as pd

import constants.column_names.silver as silver_cols
import constants.constants as const
import constants.paths as pth
from src.pipelines.utils.load_data import load_silver_data
from src.utils.logger import logger


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names by replacing spaces and special chars with underscores.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with sanitized column names.
    """
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("&", "and", regex=False)
    )
    return df


def reshape_a_and_p_data(a_and_p_data: pd.DataFrame) -> pd.DataFrame:
    """Reshape A&P data from long to wide format.

    Args:
        a_and_p_data: Input DataFrame in long format.

    Returns:
        Reshaped DataFrame in wide format with starting_week as index.
    """
    reshaped_df = (
        a_and_p_data.pivot(
            index=silver_cols.STARTING_WEEK,
            columns=silver_cols.L5,
            values=silver_cols.EXECUTION,
        )
        .fillna(0)
        .rename_axis(None, axis=1)
    )

    return reshaped_df


def create_gold_dataset(
    a_and_p_features: pd.DataFrame,
    commercial_features: pd.DataFrame,
    target_df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Create a gold dataset for MMM by merging features with a specific target.

    Args:
        a_and_p_features: Wide-format A&P features with starting_week as index.
        commercial_features: Commercial features DataFrame.
        target_df: Target DataFrame with both online and offline targets.
        target_column: Name of the target column to use.

    Returns:
        Gold dataset ready for PyMC-Marketing with starting_week as datetime index.
    """
    # Start with A&P features (already has starting_week as index)
    gold_data = a_and_p_features.copy()

    # Merge commercial features
    commercial_features_indexed = commercial_features.set_index(
        silver_cols.STARTING_WEEK
    )
    gold_data = gold_data.join(commercial_features_indexed, how="left")

    # Merge target
    target_indexed = target_df[[silver_cols.STARTING_WEEK, target_column]].set_index(
        silver_cols.STARTING_WEEK
    )
    gold_data = gold_data.join(target_indexed, how="left")

    # Rename target column to generic name
    gold_data = gold_data.rename(columns={target_column: "target"})

    # Parse index to datetime
    gold_data.index = pd.to_datetime(gold_data.index, format=const.DATE_FORMAT)
    gold_data.index.name = silver_cols.STARTING_WEEK

    # Sort by date
    gold_data = gold_data.sort_index()

    # Sanitize column names
    gold_data = sanitize_column_names(gold_data)

    return gold_data


def silver_to_gold_pipeline() -> None:
    """Transform silver data to gold data.

    Creates two gold datasets: one for online sellout, one for offline sellout.
    Each dataset is in wide format with starting_week as the datetime index.
    """
    logger.info(f"Loading silver data from {pth.SILVER_DIR}...")
    target_df, a_and_p_features, commercial_features = load_silver_data()

    logger.info("Handling duplicate L5 labels by adding L4 prefix")
    # Handling duplicate L5 entries by adding L4 prefix
    duplicated_mask = a_and_p_features[[silver_cols.L5]].duplicated(keep=False)
    a_and_p_features.loc[duplicated_mask, silver_cols.L5] = (
        a_and_p_features.loc[duplicated_mask, silver_cols.L4]
        + "_"
        + a_and_p_features.loc[duplicated_mask, silver_cols.L5]
    )

    logger.info("Reshaping A&P data to wide format")
    wide_a_and_p_features = reshape_a_and_p_data(a_and_p_features)

    # Create gold directory
    Path(pth.GOLD_DIR).mkdir(parents=True, exist_ok=True)

    # Create online gold dataset
    logger.info("Creating online gold dataset")
    online_gold = create_gold_dataset(
        a_and_p_features=wide_a_and_p_features,
        commercial_features=commercial_features,
        target_df=target_df,
        target_column=silver_cols.TARGET_ONLINE_SELLOUT_UNITS,
    )
    logger.info(f"Saving online gold dataset to {pth.GOLD_ONLINE_DATA_FILE}")
    online_gold.to_csv(pth.GOLD_ONLINE_DATA_FILE)

    # Create offline gold dataset
    logger.info("Creating offline gold dataset")
    offline_gold = create_gold_dataset(
        a_and_p_features=wide_a_and_p_features,
        commercial_features=commercial_features,
        target_df=target_df,
        target_column=silver_cols.TARGET_OFFLINE_SELLOUT_UNITS,
    )
    logger.info(f"Saving offline gold dataset to {pth.GOLD_OFFLINE_DATA_FILE}")
    offline_gold.to_csv(pth.GOLD_OFFLINE_DATA_FILE)

    logger.info("Silver to gold pipeline completed successfully")


if __name__ == "__main__":
    silver_to_gold_pipeline()
