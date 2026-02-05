"""Mappings loading functions."""

import pandas as pd

import constants.column_names.gold as gold_cols
from constants.constants import DATE_FORMAT


def load_level_mapping(level_mapping_path: str) -> pd.DataFrame:
    """Load level mapping from CSV file.

    Args:
        level_mapping_path: Path to the level mapping CSV file.

    Returns:
        DataFrame containing the level mapping.
    """
    level_mapping = pd.read_csv(level_mapping_path)
    return level_mapping


def load_spend_mapping(spend_mapping_path: str) -> pd.DataFrame:
    """Load spend mapping from CSV file.

    Args:
        spend_mapping_path: Path to the spend mapping CSV file.

    Returns:
        DataFrame containing the spend mapping.
    """
    spend_mapping = pd.read_csv(spend_mapping_path)

    # Ensure date column is in datetime format
    spend_mapping[gold_cols.DATE] = pd.to_datetime(
        spend_mapping[gold_cols.DATE], format=DATE_FORMAT
    )

    return spend_mapping
