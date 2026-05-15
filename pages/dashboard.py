# pages/dashboard.py
# The DASHBOARD page shows a visual overview of the uploaded dataset.
# It shows KPI metrics, a grid of charts, and a sample of the data.

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.express as px

from core.visualization import (
    make_histogram,
    make_scatter_plot,
    make_bar_chart,
    make_box_plot,
    make_pie_chart,
    make_violin_plot,
    make_correlation_heatmap,
)
from utils.helpers import (
    stop_if_no_data,
    get_filename,
    get_completeness,
    get_number_columns,
    get_text_columns,
)

# Stop this page from loading if no data has been uploaded yet
stop_if_no_data()

# Get the data from session state
df       = st.session_state["df"]
filename = get_filename()

# Get lists of different column types
num_cols  = get_number_columns(df)
text_cols = get_text_columns(df)

# Calculate summary numbers for the KPI row
score    = get_completeness(df)
missing  = int(df.isnull().sum().sum())
dupes    = int(df.duplicated().sum())

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 Dashboard")
st.caption(f"Viewing: **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
# Show 5 key numbers across the top of the page
st.subheader("📌 Key Numbers")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Rows",    f"{df.shape[0]:,}")
c2.metric("Columns",       df.shape[1])
c3.metric("Completeness",  f"{score}%")
c4.metric("Missing Values", missing)
c5.metric("Duplicate Rows", dupes)

# Progress bar showing how "healthy" the data is
st.progress(score / 100, text=f"Data Health: {score}%")

# ── Custom Chart Builder ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎨 Build Your Own Chart")
st.write("Choose a chart type and the columns to plot:")

# Dropdown to select chart type
chart_type = st.selectbox(
    "Chart type:",
    options=[
        "Histogram — distribution of one number column",
        "Scatter Plot — relationship between two number columns",
        "Bar Chart — count of each value in a text column",
        "Pie Chart — share of each value in a text column",
        "Box Plot — spread and outliers of number columns",
        "Violin Plot — detailed distribution of one number column",
        "Correlation Heatmap — all correlations at once",
    ]
)

# Show column selectors depending on what chart they chose
col_x = None
col_y = None

if "Histogram" in chart_type or "Violin" in chart_type:
    if num_cols:
        col_x = st.selectbox("Select column:", num_cols)
    else:
        st.warning("No numeric columns available.")

elif "Scatter" in chart_type:
    if len(num_cols) >= 2:
        col_x = st.selectbox("X-axis column:", num_cols, index=0)
        col_y = st.selectbox("Y-axis column:", num_cols, index=1)
    else:
        st.warning("Need at least 2 numeric columns for a scatter plot.")

elif "Bar" in chart_type or "Pie" in chart_type:
    if text_cols:
        col_x = st.selectbox("Select text column:", text_cols)
    else:
        st.warning("No text columns available.")

# Draw the chart when the button is clicked
# key="custom_chart" gives this chart a unique ID so Streamlit doesn't get confused
if st.button("📊 Draw Chart", use_container_width=True):
    try:
        if "Histogram" in chart_type and col_x:
            st.plotly_chart(make_histogram(df, col_x), use_container_width=True, key="custom_chart")

        elif "Scatter" in chart_type and col_x and col_y:
            st.plotly_chart(make_scatter_plot(df, col_x, col_y), use_container_width=True, key="custom_chart")

        elif "Bar" in chart_type and col_x:
            st.plotly_chart(make_bar_chart(df, col_x), use_container_width=True, key="custom_chart")

        elif "Pie" in chart_type and col_x:
            st.plotly_chart(make_pie_chart(df, col_x), use_container_width=True, key="custom_chart")

        elif "Box" in chart_type:
            st.plotly_chart(make_box_plot(df, num_cols[:6]), use_container_width=True, key="custom_chart")

        elif "Violin" in chart_type and col_x:
            st.plotly_chart(make_violin_plot(df, col_x), use_container_width=True, key="custom_chart")

        elif "Heatmap" in chart_type:
            if len(num_cols) >= 2:
                st.plotly_chart(make_correlation_heatmap(df), use_container_width=True, key="custom_chart")
            else:
                st.warning("Need at least 2 numeric columns for a heatmap.")

    except Exception as e:
        st.error(f"Could not create chart: {e}")

# ── Automatic Chart Grid ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Auto Charts")
st.write("These charts are automatically created from your data:")

# We only draw charts if there are numeric columns
if len(num_cols) >= 2:

    # Row 1: histogram and scatter side by side
    # key= gives each chart a unique name so Streamlit can tell them apart
    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.plotly_chart(make_histogram(df, num_cols[0]), use_container_width=True, key="auto_histogram")
    with row1_right:
        st.plotly_chart(make_scatter_plot(df, num_cols[0], num_cols[1]), use_container_width=True, key="auto_scatter")

    # Row 2: heatmap and bar/box chart side by side
    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.plotly_chart(make_correlation_heatmap(df), use_container_width=True, key="auto_heatmap")
    with row2_right:
        # If there's a text column, show bar chart. Otherwise, show box plot.
        if text_cols:
            st.plotly_chart(make_bar_chart(df, text_cols[0]), use_container_width=True, key="auto_bar")
        else:
            st.plotly_chart(make_box_plot(df, num_cols[:4]), use_container_width=True, key="auto_box")

    # Row 3: violin and pie/box chart
    row3_left, row3_right = st.columns(2)
    with row3_left:
        st.plotly_chart(make_violin_plot(df, num_cols[0]), use_container_width=True, key="auto_violin")
    with row3_right:
        if text_cols:
            st.plotly_chart(make_pie_chart(df, text_cols[0]), use_container_width=True, key="auto_pie")
        else:
            st.plotly_chart(make_box_plot(df, num_cols[:4]), use_container_width=True, key="auto_box2")

elif len(num_cols) == 1:
    # Only one numeric column - just show histogram and violin
    st.plotly_chart(make_histogram(df, num_cols[0]), use_container_width=True, key="single_histogram")
    st.plotly_chart(make_violin_plot(df, num_cols[0]), use_container_width=True, key="single_violin")

else:
    st.info("No numeric columns found. Cannot generate charts.")

# ── Missing Values Chart ──────────────────────────────────────────────────────
# Only show this section if there ARE missing values
missing_by_col = df.isnull().sum()
missing_by_col = missing_by_col[missing_by_col > 0]

if len(missing_by_col) > 0:
    st.markdown("---")
    st.subheader("⚠️ Missing Values by Column")

    # Build a table showing each column and its missing %
    missing_table = missing_by_col.reset_index()
    missing_table.columns = ["Column", "Missing Count"]
    missing_table["Missing %"] = (missing_table["Missing Count"] / len(df) * 100).round(1)

    # Create a bar chart of missing percentages
    fig = px.bar(
        missing_table,
        x="Column",
        y="Missing %",
        title="Missing % per Column",
        color="Missing %",
        color_continuous_scale="Reds",
        text="Missing %",
        template="plotly_dark",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True, key="missing_values_chart")

# ── Data Sample ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Random Data Sample")
st.write("10 randomly selected rows from your dataset:")

# df.sample() picks random rows
st.dataframe(df.sample(min(10, len(df))), use_container_width=True)