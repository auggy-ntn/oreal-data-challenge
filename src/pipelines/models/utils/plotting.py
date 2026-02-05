"""Plotting utilities for Bayesian MMM model visualization."""

from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import constants.column_names.contributions as contrib_cols

GranularityLevel = Literal["L2", "L3", "L4", "L5"]

LEVEL_COLUMN_MAP: dict[GranularityLevel, str] = {
    "L2": contrib_cols.L2,
    "L3": contrib_cols.L3,
    "L4": contrib_cols.L4,
    "L5": contrib_cols.L5,
}


def plot_contribution_decomposition(
    contributions: pd.DataFrame,
    granularity: GranularityLevel = "L5",
    title: str | None = None,
) -> go.Figure:
    """Plot stacked area chart of contribution decomposition by granularity level.

    Shows baseline contribution plus channel contributions aggregated at the
    specified granularity level (L2, L3, L4, or L5).

    Args:
        contributions: DataFrame with columns: starting_week, growth_driver_l2/l3/l4/l5,
            channel_contribution, baseline_contribution.
        granularity: Aggregation level - "L2", "L3", "L4", or "L5".
        title: Optional chart title. Defaults to auto-generated title.

    Returns:
        Plotly Figure object with stacked area chart.

    Example:
        >>> contributions = pd.read_csv("contributions.csv")
        >>> fig = plot_contribution_decomposition(contributions, granularity="L3")
        >>> fig.show()
    """
    level_col = LEVEL_COLUMN_MAP[granularity]

    # Aggregate channel contributions by date and granularity level
    channel_agg = (
        contributions.groupby([contrib_cols.DATE, level_col])[
            contrib_cols.CHANNEL_CONTRIBUTION
        ]
        .sum()
        .reset_index()
    )

    # Pivot to wide format for stacking
    channel_wide = channel_agg.pivot(
        index=contrib_cols.DATE,
        columns=level_col,
        values=contrib_cols.CHANNEL_CONTRIBUTION,
    ).reset_index()

    # Get baseline (one value per date, take first since they're all the same)
    baseline = (
        contributions.groupby(contrib_cols.DATE)[contrib_cols.BASELINE_CONTRIBUTION]
        .first()
        .reset_index()
    )

    # Merge baseline with channel contributions
    decomposition = pd.merge(channel_wide, baseline, on=contrib_cols.DATE)

    # Melt for plotting - baseline first, then channels
    id_vars = [contrib_cols.DATE]
    value_vars = [contrib_cols.BASELINE_CONTRIBUTION] + [
        col
        for col in decomposition.columns
        if col not in id_vars + [contrib_cols.BASELINE_CONTRIBUTION]
    ]

    plot_data = decomposition.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="component",
        value_name="contribution",
    )

    # Create stacked area chart
    fig = px.area(
        plot_data,
        x=contrib_cols.DATE,
        y="contribution",
        color="component",
        title=title or f"Sales Decomposition by {granularity} Granularity",
        labels={
            contrib_cols.DATE: "Week",
            "contribution": "Units Sold",
            "component": "Component",
        },
    )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Units Sold",
        legend_title="Component",
        hovermode="x unified",
        xaxis_tickangle=-45,
        xaxis_dtick="M1",
    )

    return fig
