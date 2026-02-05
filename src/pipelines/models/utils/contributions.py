"""Utilities for computing and visualizing channel contributions in Bayesian model."""

import pandas as pd
from pymc_marketing.mmm import MMM

import constants.column_names.contributions as contrib_cols
import constants.column_names.silver as silver_cols
import constants.paths as pth
from src.pipelines.models.utils.load_mappings import (
    load_level_mapping,
    load_spend_mapping,
)


def compute_contribution(model: MMM) -> pd.DataFrame:
    """Compute channel contributions from the trained Bayesian model.

    Args:
        model (MMM): Trained Bayesian model object.

        Returns: DataFrame containing contributions.

    Note:
        - Contributions are computed on the original target scale (native method for
        feature contributions, by hand for baseline).
    """
    # Target scaler to rescale contributions back to original target scale
    target_scaler = model.target_transformer.named_steps.get("scaler").scale_[0]

    # Intercept contribution
    intercept_contribution = (
        model.idata.posterior["intercept"].mean(dim=["chain", "draw"]).values.item()
    ) * target_scaler

    # Fourier contribution (seasonality)
    fourier_contribution = (
        model.idata.posterior["fourier_contribution"]
        .mean(dim=["chain", "draw"])
        .sum(dim="fourier_mode")
        .to_dataframe("value")
        .reset_index()[["date", "value"]]
    )
    fourier_contribution["value"] *= target_scaler

    # Control variable contribution
    control_contribution = (
        model.idata.posterior["control_contribution"]
        .mean(dim=["chain", "draw"])
        .sum(dim="control")
        .to_dataframe("value")
        .reset_index()[["date", "value"]]
    )
    control_contribution["value"] *= target_scaler

    baseline_contribution = fourier_contribution.copy()
    baseline_contribution["value"] += (
        intercept_contribution + control_contribution["value"]
    )
    baseline_contribution = baseline_contribution.rename(
        columns={"value": contrib_cols.BASELINE_CONTRIBUTION}
    )

    # Channel contributions
    contributions = model.compute_channel_contribution_original_scale()

    # One contribution per (date, channel) by averaging over chains and draws
    mean_channel = (
        contributions.mean(dim=["chain", "draw"]).to_dataframe("value").reset_index()
    )

    # Rename columns
    mean_channel = mean_channel.rename(
        columns={
            "date": contrib_cols.DATE,
            "channel": contrib_cols.L5,
            "value": contrib_cols.CHANNEL_CONTRIBUTION,
        }
    )

    # Merge with level mapping to get L4, L3 and L2 information
    level_mapping = load_level_mapping(pth.GOLD_LEVEL_MAPPING_FILE)
    mean_channel = pd.merge(
        mean_channel,
        level_mapping,
        on=contrib_cols.L5,
        how="left",
        validate="m:1",
    )

    # Merge with spend mapping to get spend information for ROI calculation
    spend_mapping = load_spend_mapping(pth.GOLD_SPEND_MAPPING_FILE)
    mean_channel = pd.merge(
        mean_channel,
        spend_mapping.rename(
            columns={silver_cols.INVESTMENT: contrib_cols.SPEND},
        ),
        left_on=[contrib_cols.DATE, contrib_cols.L5],
        right_on=[contrib_cols.DATE, silver_cols.L5],
        how="left",
        validate="1:1",
    )

    # Reorder columns
    mean_channel = mean_channel[
        [
            contrib_cols.DATE,
            contrib_cols.L2,
            contrib_cols.L3,
            contrib_cols.L4,
            contrib_cols.L5,
            contrib_cols.CHANNEL_CONTRIBUTION,
            contrib_cols.SPEND,
        ]
    ]

    contributions_df = pd.merge(
        mean_channel,
        baseline_contribution.rename(columns={"date": contrib_cols.DATE}),
        on=contrib_cols.DATE,
        how="left",
        validate="m:1",
    )

    return contributions_df
