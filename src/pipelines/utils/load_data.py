"""Data loading utilities for data."""

import pandas as pd

import constants.paths as pth


def load_bronze_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bronze data from an Excel file.

    Returns:
        tuple: A tuple containing two DataFrames:
            - target_df: DataFrame with KPI data.
            - a_and_p_df: DataFrame with A&P variables data.
    """
    target_df = pd.read_excel(pth.BRONZE_DATA_FILE, sheet_name=pth.KPI_SHEET)
    a_and_p_df = pd.read_excel(pth.BRONZE_DATA_FILE, sheet_name=pth.A_AND_P_SHEET)
    return target_df, a_and_p_df


def load_silver_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load silver data from CSV files.

    Returns:
        tuple: A tuple containing two DataFrames:
            - target_df: DataFrame with target data.
            - a_and_p_df: DataFrame with A&P features data.
    """
    target_df = pd.read_csv(pth.SILVER_TARGET_FILE)
    a_and_p_df = pd.read_csv(pth.SILVER_A_AND_P_FEATURES_FILE)
    return target_df, a_and_p_df
