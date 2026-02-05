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


# Color palette for categories (matching the image style)
CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    # (lighter shade for year 1, darker shade for year 2)
    "default": ("#90EE90", "#228B22"),  # Green shades
    "consumer_engagement": ("#90EE90", "#228B22"),  # Green
    "paid_media": ("#90EE90", "#228B22"),  # Green
    "advocacy_media": ("#FFE4B5", "#DAA520"),  # Yellow/Gold
    "owned_media": ("#87CEEB", "#1E90FF"),  # Blue
    "shopper_experience": ("#FFB6C1", "#CD5C5C"),  # Red/Pink
}


def _prepare_yearly_data(
    contributions: pd.DataFrame,
    granularity: GranularityLevel,
) -> tuple[pd.DataFrame, list[int], list[str], str]:
    """Prepare aggregated yearly data for plotting.

    Args:
        contributions: Raw contributions DataFrame.
        granularity: Aggregation level.

    Returns:
        Tuple of (aggregated DataFrame, years list, categories list, level column name).
    """
    level_col = LEVEL_COLUMN_MAP[granularity]

    # Extract year from date
    df = contributions.copy()
    df[contrib_cols.DATE] = pd.to_datetime(df[contrib_cols.DATE])
    df["year"] = df[contrib_cols.DATE].dt.year

    # Aggregate by year and granularity level
    yearly_agg = (
        df.groupby(["year", level_col])
        .agg(
            {
                contrib_cols.CHANNEL_CONTRIBUTION: "sum",
                contrib_cols.SPEND: "sum",
            }
        )
        .reset_index()
    )

    # Calculate ROI (handle division by zero)
    yearly_agg["roi"] = yearly_agg[contrib_cols.CHANNEL_CONTRIBUTION] / yearly_agg[
        contrib_cols.SPEND
    ].replace(0, float("nan"))

    # Also compute total media (sum across all categories per year)
    total_by_year = (
        df.groupby("year")
        .agg(
            {
                contrib_cols.CHANNEL_CONTRIBUTION: "sum",
                contrib_cols.SPEND: "sum",
            }
        )
        .reset_index()
    )
    total_by_year[level_col] = "TOTAL MEDIA"
    total_by_year["roi"] = total_by_year[
        contrib_cols.CHANNEL_CONTRIBUTION
    ] / total_by_year[contrib_cols.SPEND].replace(0, float("nan"))

    # Combine total with category data
    yearly_agg = pd.concat([total_by_year, yearly_agg], ignore_index=True)

    # Get unique years and categories
    years = sorted(yearly_agg["year"].unique())
    categories = yearly_agg[level_col].unique().tolist()

    # Ensure TOTAL MEDIA is first
    if "TOTAL MEDIA" in categories:
        categories.remove("TOTAL MEDIA")
        categories = ["TOTAL MEDIA"] + categories

    return yearly_agg, years, categories, level_col


def _create_yearly_bar_chart(
    yearly_agg: pd.DataFrame,
    years: list[int],
    categories: list[str],
    level_col: str,
    value_col: str,
    title: str,
    x_axis_title: str,
    value_format: str = ",.0f",
) -> go.Figure:
    """Create a horizontal bar chart for yearly comparison.

    Args:
        yearly_agg: Aggregated data.
        years: List of years.
        categories: List of categories.
        level_col: Column name for granularity level.
        value_col: Column to plot.
        title: Chart title.
        x_axis_title: X-axis label.
        value_format: Format string for bar labels.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Color mapping for years
    year_opacity = {years[i]: 0.6 if i == 0 else 1.0 for i in range(len(years))}

    # Add bars for each category and year
    for cat_idx, category in enumerate(reversed(categories)):
        cat_data = yearly_agg[yearly_agg[level_col] == category]

        # Get color for this category
        cat_key = category.lower().replace(" ", "_")
        colors = CATEGORY_COLORS.get(cat_key, CATEGORY_COLORS["default"])

        for year_idx, year in enumerate(years):
            year_data = cat_data[cat_data["year"] == year]
            if year_data.empty:
                continue

            color = colors[min(year_idx, len(colors) - 1)]
            show_legend = cat_idx == 0

            value = year_data[value_col].values[0]
            if pd.isna(value):
                text = "-"
                value = 0
            else:
                text = f"{value:{value_format}}"

            fig.add_trace(
                go.Bar(
                    y=[category],
                    x=[value],
                    orientation="h",
                    name=str(year),
                    marker_color=color,
                    opacity=year_opacity[year],
                    text=[text],
                    textposition="outside",
                    textfont=dict(size=10),
                    cliponaxis=False,
                    showlegend=show_legend,
                    legendgroup=str(year),
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#20B2AA", size=16)),
        barmode="group",
        height=150 + len(categories) * 80,
        width=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title="Year",
        ),
        margin=dict(l=200, r=150),
        xaxis_title=x_axis_title,
    )

    return fig


def plot_yearly_incremental_units(
    contributions: pd.DataFrame,
    granularity: GranularityLevel = "L2",
    title: str = "INCREMENTAL UNITS",
) -> go.Figure:
    """Plot yearly incremental units (channel contributions) by granularity level.

    Args:
        contributions: DataFrame with contributions data.
        granularity: Aggregation level - "L2", "L3", "L4", or "L5".
        title: Chart title.

    Returns:
        Plotly Figure with horizontal bar chart.
    """
    yearly_agg, years, categories, level_col = _prepare_yearly_data(
        contributions, granularity
    )
    return _create_yearly_bar_chart(
        yearly_agg=yearly_agg,
        years=years,
        categories=categories,
        level_col=level_col,
        value_col=contrib_cols.CHANNEL_CONTRIBUTION,
        title=title,
        x_axis_title="Units",
        value_format=",.0f",
    )


def plot_yearly_spend(
    contributions: pd.DataFrame,
    granularity: GranularityLevel = "L2",
    title: str = "SPEND",
) -> go.Figure:
    """Plot yearly spend by granularity level.

    Args:
        contributions: DataFrame with contributions data.
        granularity: Aggregation level - "L2", "L3", "L4", or "L5".
        title: Chart title.

    Returns:
        Plotly Figure with horizontal bar chart.
    """
    yearly_agg, years, categories, level_col = _prepare_yearly_data(
        contributions, granularity
    )
    return _create_yearly_bar_chart(
        yearly_agg=yearly_agg,
        years=years,
        categories=categories,
        level_col=level_col,
        value_col=contrib_cols.SPEND,
        title=title,
        x_axis_title="Currency",
        value_format=",.0f",
    )


def plot_yearly_roi(
    contributions: pd.DataFrame,
    granularity: GranularityLevel = "L2",
    title: str = "ROI",
) -> go.Figure:
    """Plot yearly ROI (contribution / spend) by granularity level.

    Args:
        contributions: DataFrame with contributions data.
        granularity: Aggregation level - "L2", "L3", "L4", or "L5".
        title: Chart title.

    Returns:
        Plotly Figure with horizontal bar chart.
    """
    yearly_agg, years, categories, level_col = _prepare_yearly_data(
        contributions, granularity
    )
    return _create_yearly_bar_chart(
        yearly_agg=yearly_agg,
        years=years,
        categories=categories,
        level_col=level_col,
        value_col="roi",
        title=title,
        x_axis_title="Ratio",
        value_format=".2f",
    )
