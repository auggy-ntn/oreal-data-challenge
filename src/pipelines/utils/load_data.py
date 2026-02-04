"""Data loading utilities for data."""

import pandas as pd

import constants.column_names.gold as gold_cols
from constants.constants import DATE_FORMAT
import constants.paths as pth


def load_bronze_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load bronze data from an Excel file.

    Returns:
        tuple: A tuple containing three DataFrames:
            - target_df: DataFrame with KPI data.
            - a_and_p_df: DataFrame with A&P variables data.
            - commercial_df: DataFrame with commercial data (control variables).

    """
    target_df = pd.read_excel(pth.BRONZE_DATA_FILE, sheet_name=pth.KPI_SHEET)
    a_and_p_df = pd.read_excel(pth.BRONZE_DATA_FILE, sheet_name=pth.A_AND_P_SHEET)
    commercial_df = pd.read_excel(pth.BRONZE_DATA_FILE, sheet_name=pth.COMMERCIAL_SHEET)
    return target_df, a_and_p_df, commercial_df


def load_silver_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load silver data from CSV files.

    Returns:
        tuple: A tuple containing three DataFrames:
            - target_df: DataFrame with target data.
            - a_and_p_df: DataFrame with A&P features data.
            - commercial_df: DataFrame with commercial features data.
    """
    target_df = pd.read_csv(pth.SILVER_TARGET_FILE)
    a_and_p_df = pd.read_csv(pth.SILVER_A_AND_P_FEATURES_FILE)
    commercial_df = pd.read_csv(pth.SILVER_COMMERCIAL_FEATURES_FILE)
    return target_df, a_and_p_df, commercial_df


def load_gold_data(online: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Load gold data from a CSV file.

    Returns:
        tuple: A tuple containing:
            - X: DataFrame with features.
            - y: Series with target variable.
    """
    gold_df = pd.read_csv(
        pth.GOLD_ONLINE_DATA_FILE if online else pth.GOLD_OFFLINE_DATA_FILE
    )
    X = gold_df.drop(columns=["target"])
    X[gold_cols.DATE] = pd.to_datetime(X[gold_cols.DATE], format=DATE_FORMAT)
    y = gold_df["target"]
    return X, y
