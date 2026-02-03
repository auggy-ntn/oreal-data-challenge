"""Streamlit Dashboard for L'Oréal MMM Analysis.

This dashboard provides interactive exploration of:
- Model Results & Diagnostics
- Channel Attribution & ROI
- Saturation Analysis
- Optimization Recommendations
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.data.data_loader import load_unified_dataset  # noqa: E402
from src.pipelines.models.transformations import (  # noqa: E402
    get_default_params,
    get_saturation_curve_points,
    hill_saturation,
    transform_all_channels,
)

st.set_page_config(
    page_title="L'Oréal MMM Dashboard",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3436;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load and cache the unified dataset."""
    df = load_unified_dataset()
    df = transform_all_channels(df)
    return df


@st.cache_data
def load_results():
    """Load pre-computed results from CSV files."""
    output_dir = project_root / "outputs"

    results = {}

    # Load insights
    insights_dir = output_dir / "insights"
    if insights_dir.exists():
        for file in insights_dir.glob("*.csv"):
            name = file.stem
            results[name] = pd.read_csv(file)

    return results


def render_sidebar():
    """Render the sidebar with navigation and filters."""
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/L%27Or%C3%A9al_logo.svg/200px-L%27Or%C3%A9al_logo.svg.png",
        width=150,
    )
    st.sidebar.title("MMM Dashboard")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Overview",
            "📈 Model Results",
            "💰 ROI Analysis",
            "📉 Saturation Curves",
            "🎯 Optimization",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**L'Oréal Paris Haircare UK**")
    st.sidebar.markdown("*BETiq MMM Analysis*")

    return page


def render_overview(df: pd.DataFrame, results: dict):
    """Render the overview page."""
    st.markdown(
        '<p class="main-header">🎯 Marketing Mix Modeling Dashboard</p>',
        unsafe_allow_html=True,
    )

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_offline = df["offline_units"].sum() / 1e6
        st.metric("Total Offline Units", f"{total_offline:.1f}M", "2-year total")

    with col2:
        total_online = df["online_units"].sum() / 1e6
        st.metric("Total Online Units", f"{total_online:.1f}M", "2-year total")

    with col3:
        total_investment = (
            df[[c for c in df.columns if "_investment" in c]].sum().sum() / 1e6
        )
        st.metric("Total A&P Investment", f"£{total_investment:.1f}M", "2-year total")

    with col4:
        weeks = len(df)
        st.metric("Data Points", f"{weeks} weeks", "Jan 2022 - Dec 2023")

    st.markdown("---")

    # Time Series
    st.markdown(
        '<p class="section-header">📈 Sales Time Series</p>', unsafe_allow_html=True
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["week"],
            y=df["offline_units"],
            name="Offline Units",
            line=dict(color="#667eea", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["week"],
            y=df["online_units"],
            name="Online Units",
            line=dict(color="#f093fb", width=2),
        )
    )

    fig.update_layout(
        title="Weekly Sales Performance",
        xaxis_title="Week",
        yaxis_title="Units",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Investment by Channel
    st.markdown(
        '<p class="section-header">💵 A&P Investment Distribution</p>',
        unsafe_allow_html=True,
    )

    investment_cols = [c for c in df.columns if c.endswith("_investment")]
    investment_totals = df[investment_cols].sum()
    investment_totals.index = [
        c.replace("_investment", "").replace("_", " ").title()
        for c in investment_totals.index
    ]
    investment_totals = investment_totals.sort_values(ascending=False)

    fig2 = px.bar(
        x=investment_totals.values / 1e6,
        y=investment_totals.index,
        orientation="h",
        labels={"x": "Investment (£M)", "y": "Channel"},
        color=investment_totals.values,
        color_continuous_scale="Viridis",
    )
    fig2.update_layout(
        title="Total Investment by Channel",
        showlegend=False,
        template="plotly_white",
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_model_results(df: pd.DataFrame, results: dict):
    """Render model results page."""
    st.markdown('<p class="main-header">📈 Model Results</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Offline Units Model")
        st.metric("R²", "0.711", "71.1% variance explained")
        st.metric("Adj. R²", "0.613")
        st.metric("Durbin-Watson", "2.22", "No autocorrelation")

        # Load image if exists
        diag_path = (
            project_root / "outputs" / "models" / "diagnostics_offline_units.png"
        )
        if diag_path.exists():
            st.image(str(diag_path), caption="Offline Model Diagnostics")

    with col2:
        st.markdown("### Online Units Model")
        st.metric("R²", "0.837", "83.7% variance explained")
        st.metric("Adj. R²", "0.781")
        st.metric("Durbin-Watson", "1.81", "Acceptable range")

        # Load image if exists
        diag_path = project_root / "outputs" / "models" / "diagnostics_online_units.png"
        if diag_path.exists():
            st.image(str(diag_path), caption="Online Model Diagnostics")


def render_roi_analysis(df: pd.DataFrame, results: dict):
    """Render ROI analysis page."""
    st.markdown('<p class="main-header">💰 ROI Analysis</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Offline Channel", "Online Channel"])

    with tab1:
        if "offline_roi" in results:
            roi_df = results["offline_roi"]

            # Bubble chart
            fig = px.scatter(
                roi_df[roi_df["investment"] > 0],
                x="roi",
                y="due_to_units",
                size="investment",
                color="significant",
                hover_name="channel",
                color_discrete_map={True: "green", False: "gray"},
                labels={
                    "roi": "ROI",
                    "due_to_units": "Incremental Units",
                    "investment": "Investment",
                },
            )
            fig.add_vline(
                x=1, line_dash="dash", line_color="blue", annotation_text="Break-even"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Offline: ROI vs Contribution", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(
                roi_df[["channel", "investment", "due_to_units", "roi", "significant"]],
                use_container_width=True,
            )
        else:
            st.warning("Run the insights pipeline to generate ROI data")

    with tab2:
        if "online_roi" in results:
            roi_df = results["online_roi"]

            fig = px.scatter(
                roi_df[roi_df["investment"] > 0],
                x="roi",
                y="due_to_units",
                size="investment",
                color="significant",
                hover_name="channel",
                color_discrete_map={True: "green", False: "gray"},
            )
            fig.add_vline(
                x=1, line_dash="dash", line_color="blue", annotation_text="Break-even"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Online: ROI vs Contribution", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                roi_df[["channel", "investment", "due_to_units", "roi", "significant"]],
                use_container_width=True,
            )
        else:
            st.warning("Run the insights pipeline to generate ROI data")


def render_saturation_curves():
    """Render saturation curves page."""
    st.markdown(
        '<p class="main-header">📉 Saturation Analysis</p>', unsafe_allow_html=True
    )

    channels = [
        "linear",
        "bvod",
        "meta",
        "tik_tok",
        "pinterest",
        "youtube",
        "google",
        "amazon",
        "tesco",
    ]

    selected_channel = st.selectbox(
        "Select Channel",
        channels,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    params = get_default_params(selected_channel)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Interactive saturation curve
        x = np.linspace(0, 2, 100)
        y = hill_saturation(x, params["saturation_k"], params["saturation_s"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Saturation Curve",
                line=dict(color="#667eea", width=3),
            )
        )

        # Add ABCD points
        abcd = get_saturation_curve_points(
            params["saturation_k"], params["saturation_s"]
        )
        for name, (px_val, py_val) in abcd.items():
            if px_val <= 2:
                fig.add_trace(
                    go.Scatter(
                        x=[px_val],
                        y=[py_val],
                        mode="markers+text",
                        name=name,
                        text=[name],
                        textposition="top right",
                        marker=dict(size=12),
                    )
                )

        fig.update_layout(
            title=f"Saturation Curve: {selected_channel.replace('_', ' ').title()}",
            xaxis_title="Normalized Execution",
            yaxis_title="Effect (%)",
            template="plotly_white",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Parameters")
        st.metric("Adstock Decay", f"{params['adstock']:.2f}")
        st.metric("Saturation K", f"{params['saturation_k']:.2f}")
        st.metric("Saturation S", f"{params['saturation_s']:.2f}")
        st.metric("Lag (weeks)", f"{params['lag']}")

        st.markdown("### ABCD Points")
        for name, (px_val, py_val) in abcd.items():
            st.write(f"**{name}**: x={px_val:.2f}, y={py_val:.0%}")


def render_optimization(df: pd.DataFrame, results: dict):
    """Render optimization recommendations page."""
    st.markdown(
        '<p class="main-header">🎯 Optimization Recommendations</p>',
        unsafe_allow_html=True,
    )

    if "recommendations" in results:
        rec_df = results["recommendations"]

        # Split by recommendation type
        priority_df = rec_df[
            rec_df["recommendation"].isin(
                ["Significantly Increase", "Reduce/Reallocate"]
            )
        ]

        st.markdown("### 🚨 Priority Actions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Increase Investment")
            increase_df = priority_df[
                priority_df["recommendation"] == "Significantly Increase"
            ]
            for _, row in increase_df.iterrows():
                channel_name = row["channel"].replace("_", " ").title()
                target = row["target_channel"].title()
                st.success(f"**{channel_name}** ({target}) - ROI: {row['roi']:.1f}x")

        with col2:
            st.markdown("#### 📉 Reduce/Reallocate")
            reduce_df = priority_df[
                priority_df["recommendation"] == "Reduce/Reallocate"
            ]
            for _, row in reduce_df.iterrows():
                channel_name = row["channel"].replace("_", " ").title()
                target = row["target_channel"].title()
                st.error(f"**{channel_name}** ({target}) - ROI: {row['roi']:.1f}x")

        st.markdown("---")
        st.markdown("### Full Recommendations Table")
        st.dataframe(rec_df, use_container_width=True)

    else:
        st.warning("Run the insights pipeline to generate recommendations")


def main():
    """Main dashboard function."""
    page = render_sidebar()

    try:
        df = load_data()
        results = load_results()

        if page == "📊 Overview":
            render_overview(df, results)
        elif page == "📈 Model Results":
            render_model_results(df, results)
        elif page == "💰 ROI Analysis":
            render_roi_analysis(df, results)
        elif page == "📉 Saturation Curves":
            render_saturation_curves()
        elif page == "🎯 Optimization":
            render_optimization(df, results)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info(
            "Make sure to run the MMM pipeline first: "
            "`uv run python -m src.pipelines.models.insights`"
        )


if __name__ == "__main__":
    main()
