# core/visualization.py
# All chart functions using Plotly.
# FIX: Every chart uses template="plotly_dark" but renderer is SVG-compatible
# to avoid "WebGL not supported" blank chart errors on Streamlit Cloud.

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Force SVG renderer globally — fixes blank/WebGL charts on Streamlit Cloud
pio.renderers.default = "svg"


def _dark_layout():
    """Common dark layout settings applied to every chart."""
    return dict(
        paper_bgcolor="#161b25",
        plot_bgcolor="#0d0f14",
        font=dict(color="#c8d0e0", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
    )


def make_histogram(df, column):
    """Distribution of a numeric column."""
    fig = px.histogram(
        df, x=column, nbins=40,
        title=f"Distribution of {column}",
        color_discrete_sequence=["#4f8ef7"],
    )
    fig.update_layout(**_dark_layout(), bargap=0.05)
    fig.update_xaxes(gridcolor="#1e2535")
    fig.update_yaxes(gridcolor="#1e2535")
    return fig


def make_scatter_plot(df, x_column, y_column):
    """Relationship between two numeric columns."""
    fig = px.scatter(
        df, x=x_column, y=y_column,
        title=f"{x_column} vs {y_column}",
        opacity=0.6,
        color_discrete_sequence=["#4f8ef7"],
    )
    fig.update_layout(**_dark_layout())
    fig.update_xaxes(gridcolor="#1e2535")
    fig.update_yaxes(gridcolor="#1e2535")
    return fig


def make_bar_chart(df, column, top_n=10):
    """Count of top N values in a text column."""
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = ["Value", "Count"]
    fig = px.bar(
        counts, x="Count", y="Value",
        orientation="h",
        title=f"Top {top_n} values in '{column}'",
        color="Count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(**_dark_layout())
    fig.update_xaxes(gridcolor="#1e2535")
    return fig


def make_box_plot(df, columns):
    """Spread and outliers of numeric columns."""
    if isinstance(columns, list) and len(columns) == 0:
        columns = df.select_dtypes(include="number").columns.tolist()
    if not columns:
        fig = go.Figure()
        fig.update_layout(title="Box Plot — No numeric columns", **_dark_layout())
        return fig
    fig = px.box(
        df, y=columns,
        title="Box Plot — Spread and Outliers",
        color_discrete_sequence=["#4f8ef7"],
    )
    fig.update_layout(**_dark_layout())
    fig.update_xaxes(gridcolor="#1e2535")
    fig.update_yaxes(gridcolor="#1e2535")
    return fig


def make_violin_plot(df, column):
    """Full distribution shape of a numeric column."""
    fig = px.violin(
        df, y=column,
        box=True, points="outliers",
        title=f"Violin Plot of {column}",
        color_discrete_sequence=["#7c5cfc"],
    )
    fig.update_layout(**_dark_layout())
    fig.update_yaxes(gridcolor="#1e2535")
    return fig


def make_correlation_heatmap(df):
    """Correlation between all numeric columns."""
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) < 2:
        fig = go.Figure()
        fig.update_layout(title="Need at least 2 numeric columns", **_dark_layout())
        return fig
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    fig.update_layout(**_dark_layout())
    return fig


def make_pie_chart(df, column, top_n=8):
    """Share of each category in a text column."""
    counts = df[column].value_counts().head(top_n)
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title=f"Share of values in '{column}'",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_layout(**_dark_layout())
    return fig


def get_auto_charts(df):
    """Automatically creates the best charts for this DataFrame."""
    charts   = {}
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    if len(num_cols) >= 1:
        charts["Histogram"] = make_histogram(df, num_cols[0])
        charts["Violin"]    = make_violin_plot(df, num_cols[0])
        charts["Box Plot"]  = make_box_plot(df, num_cols[:6])

    if len(num_cols) >= 2:
        charts["Scatter"] = make_scatter_plot(df, num_cols[0], num_cols[1])
        charts["Heatmap"] = make_correlation_heatmap(df)

    if len(text_cols) >= 1:
        charts["Bar Chart"] = make_bar_chart(df, text_cols[0])
        charts["Pie Chart"] = make_pie_chart(df, text_cols[0])

    return charts