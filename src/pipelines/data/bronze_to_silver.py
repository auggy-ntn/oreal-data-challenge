"""Bronze to Silver pipeline."""

from pathlib import Path

import constants.column_names.bronze as bronze_cols
import constants.column_names.silver as silver_cols
import constants.paths as pth
from src.pipelines.utils.load_data import load_bronze_data
from src.utils.logger import logger


def bronze_to_silver_pipeline() -> None:
    """Transform bronze data to silver data.

    Loads the bronze data and saves the transformed data as silver data.
    """
    logger.info(f"Loading bronze data from {pth.BRONZE_DATA_FILE}...")
    target_df, a_and_p_df, commercial_df = load_bronze_data()

    # Rename columns for target DataFrame
    target_column_mapping = {
        bronze_cols.STARTING_WEEK: silver_cols.STARTING_WEEK,
        bronze_cols.YEAR: silver_cols.YEAR,
        bronze_cols.TARGET_ONLINE_SELLOUT_UNITS: (
            silver_cols.TARGET_ONLINE_SELLOUT_UNITS
        ),
        bronze_cols.TARGET_OFFLINE_SELLOUT_UNITS: (
            silver_cols.TARGET_OFFLINE_SELLOUT_UNITS
        ),
    }
    target_df.columns = target_df.columns.str.lower()  # Standardize to lowercase
    target_df = target_df.rename(columns=target_column_mapping)
    # We only perform MMM on sellout units
    target_df = target_df[
        [
            silver_cols.STARTING_WEEK,
            silver_cols.TARGET_ONLINE_SELLOUT_UNITS,
            silver_cols.TARGET_OFFLINE_SELLOUT_UNITS,
        ]
    ]

    # Rename columns for A&P DataFrame
    a_and_p_column_mapping = {
        bronze_cols.STARTING_WEEK: silver_cols.STARTING_WEEK,
        bronze_cols.YEAR: silver_cols.YEAR,
        bronze_cols.L1: silver_cols.L1,
        bronze_cols.L2: silver_cols.L2,
        bronze_cols.L3: silver_cols.L3,
        bronze_cols.L4: silver_cols.L4,
        bronze_cols.L5: silver_cols.L5,
        bronze_cols.INVESTMENT: silver_cols.INVESTMENT,
        bronze_cols.EXECUTION: silver_cols.EXECUTION,
    }
    a_and_p_df.columns = a_and_p_df.columns.str.lower()  # Standardize to lowercase
    a_and_p_df = a_and_p_df.rename(columns=a_and_p_column_mapping)

    # Rename columns for Commercial DataFrame
    commercial_column_mapping = {
        bronze_cols.STARTING_WEEK: silver_cols.STARTING_WEEK,
        bronze_cols.YEAR: silver_cols.YEAR,
        bronze_cols.OFFLINE_AVG_PRICE: silver_cols.OFFLINE_AVG_PRICE,
        bronze_cols.ONLINE_AVG_PRICE: silver_cols.ONLINE_AVG_PRICE,
        bronze_cols.TOTAL_WEIGHTED_PROMOTION_DISTRIBUTION: (
            silver_cols.TOTAL_WEIGHTED_PROMOTION_DISTRIBUTION
        ),
    }
    commercial_df.columns = (
        commercial_df.columns.str.lower()
    )  # Standardize to lowercase
    commercial_df = commercial_df.rename(columns=commercial_column_mapping)
    commercial_df = commercial_df[
        [
            silver_cols.STARTING_WEEK,
            silver_cols.OFFLINE_AVG_PRICE,
            silver_cols.ONLINE_AVG_PRICE,
            silver_cols.TOTAL_WEIGHTED_PROMOTION_DISTRIBUTION,
        ]
    ]

    # Save silver data
    Path(pth.SILVER_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving silver target data to {pth.SILVER_TARGET_FILE}...")
    target_df.to_csv(pth.SILVER_TARGET_FILE, index=False)
    logger.info(
        f"Saving silver A&P features data to {pth.SILVER_A_AND_P_FEATURES_FILE}..."
    )
    a_and_p_df.to_csv(pth.SILVER_A_AND_P_FEATURES_FILE, index=False)
    logger.info(
        "Saving silver commercial features data to "
        f"{pth.SILVER_COMMERCIAL_FEATURES_FILE}..."
    )
    commercial_df.to_csv(pth.SILVER_COMMERCIAL_FEATURES_FILE, index=False)


if __name__ == "__main__":
    bronze_to_silver_pipeline()
