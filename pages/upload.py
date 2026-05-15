# pages/upload.py
# This is the UPLOAD page - the first page users visit.
# It lets users upload a file, see what's in it, clean the data,
# view charts, get AI insights, and detect anomalies.

# These 2 lines fix the import paths so Python can find our core/ and utils/ folders
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# Import our own modules
from core.data_loader       import load_file, get_basic_info
from core.data_cleaner      import (
    get_quality_report,
    remove_duplicates,
    drop_rows_with_missing,
    fill_with_mean,
    fill_with_median,
    fill_with_mode,
    fill_with_custom_value,
    drop_high_missing_columns,
    auto_fix_types,
)
from core.visualization     import get_auto_charts
from core.insight_generator import generate_insights
from core.anomaly_detector  import detect_anomalies
from utils.helpers          import (
    save_dataframe,
    get_filename,
    get_completeness,
)

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📂 Upload & Clean")
st.write("Upload your data file, clean it up, then explore using the other pages.")
st.markdown("---")

# ── Step 1: File Upload ───────────────────────────────────────────────────────
st.subheader("Step 1 — Upload your file")

# st.file_uploader shows a drag-and-drop upload box
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls", "json"],  # only allow these file types
    help="Supported formats: CSV, Excel (.xlsx or .xls), JSON",
)

# If the user uploaded something, try to load it
if uploaded_file is not None:

    # ── THE BUG FIX ───────────────────────────────────────────────────────────
    # PROBLEM: Streamlit reruns the entire page after every button click.
    # Every rerun, uploaded_file is still not None (file is still in the widget).
    # Without this check, every cleaning action would reload the original file
    # and wipe out all your changes!
    #
    # FIX: Only reload the file if the filename is DIFFERENT from what we
    # already have loaded. If it's the same file, skip loading and keep
    # the current (possibly cleaned) version in session state.

    already_loaded = st.session_state.get("filename", "") == uploaded_file.name

    if not already_loaded:
        # This is a NEW file — load it fresh
        try:
            df = load_file(uploaded_file)

            # Save to session state so ALL pages can access the same data
            st.session_state["df"]          = df
            st.session_state["df_original"] = df.copy()  # backup for reset button
            st.session_state["filename"]    = uploaded_file.name

            st.success(f"✅ File **{uploaded_file.name}** loaded successfully!")

        except Exception as error:
            st.error(f"❌ Could not read the file: {error}")
            st.stop()
    # If already_loaded is True, we do nothing — keep the cleaned df as-is

# If no data is in session state yet, stop here and show a message
if "df" not in st.session_state:
    st.info("👆 Upload a file above to get started.")
    st.stop()

# From here on, we know a file has been uploaded
df       = st.session_state["df"]        # the current (possibly cleaned) data
filename = get_filename()                  # the uploaded file's name
info     = get_basic_info(df)             # basic stats dictionary

# ── Step 2: Overview ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 2 — Overview")

# st.columns(5) creates 5 side-by-side boxes
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows",        f"{info['rows']:,}")        # :, adds thousands separator
c2.metric("Columns",     info["columns"])
c3.metric("Numeric",     len(info["numeric"]))
c4.metric("Categorical", len(info["categorical"]))
c5.metric("Missing",     info["missing"])

# Show first 10 rows of the data
st.write("**First 10 rows of your data:**")
st.dataframe(df.head(10), use_container_width=True)

# ── Step 3: Data Quality ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 3 — Data Quality")

report = get_quality_report(df)
score  = get_completeness(df)

# Show a progress bar for completeness score
st.progress(score / 100, text=f"Data completeness: {score}%")

# Show missing values in two columns side by side
left_col, right_col = st.columns(2)

with left_col:
    if len(report["missing_per_column"]) == 0:
        st.success("✅ No missing values found!")
    else:
        st.warning(f"⚠️ Missing values in {len(report['missing_per_column'])} columns:")
        # Convert to a neat table to display
        missing_table = report["missing_per_column"].reset_index()
        missing_table.columns = ["Column", "Missing Count"]
        missing_table["Missing %"] = (missing_table["Missing Count"] / len(df) * 100).round(1)
        st.dataframe(missing_table, use_container_width=True, hide_index=True)

with right_col:
    if report["duplicate_count"] == 0:
        st.success("✅ No duplicate rows found!")
    else:
        st.warning(f"⚠️ Found {report['duplicate_count']} duplicate rows")

# ── Step 4: Clean the Data ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 4 — Clean the Data")
st.write("Click any action below to clean your data. Changes apply immediately.")

# st.tabs creates a tab bar with multiple tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🩹 Fix Missing Values",
    "🗑️ Remove Duplicates",
    "✂️ Drop Columns",
    "🔧 Fix Data Types",
])

# Tab 1 — Fix missing values
with tab1:
    st.write("Choose how to fill or remove missing values:")

    # st.radio shows a list of options the user can pick one from
    strategy = st.radio(
        "Pick a strategy:",
        options=[
            "Fill with Mean (average) — good for normal data",
            "Fill with Median (middle value) — better for skewed data",
            "Fill with Mode (most common value) — works for text too",
            "Fill with a Custom Value — you choose what to fill with",
            "Drop rows that have any missing value",
            "Drop columns that are mostly empty",
        ],
    )

    # Show extra input only if certain strategies are chosen
    custom_value = None
    if "Custom" in strategy:
        custom_value = st.text_input("Enter the value to fill with:", "0")

    threshold = 50  # default threshold
    if "mostly empty" in strategy:
        threshold = st.slider(
            "Drop columns missing more than this % of values:",
            min_value=10, max_value=90, value=50
        )

    # Apply button
    if st.button("✅ Apply This Strategy", use_container_width=True):
        if "Mean" in strategy:
            df, count = fill_with_mean(df)
            st.success(f"Done! Filled {count} empty cells with the mean.")

        elif "Median" in strategy:
            df, count = fill_with_median(df)
            st.success(f"Done! Filled {count} empty cells with the median.")

        elif "Mode" in strategy:
            df, count = fill_with_mode(df)
            st.success(f"Done! Filled {count} empty cells with the mode.")

        elif "Custom" in strategy:
            df, count = fill_with_custom_value(df, custom_value)
            st.success(f"Done! Filled {count} empty cells with '{custom_value}'.")

        elif "Drop rows" in strategy:
            df, count = drop_rows_with_missing(df)
            st.success(f"Done! Removed {count} rows that had empty values.")

        elif "mostly empty" in strategy:
            df, dropped_cols = drop_high_missing_columns(df, threshold)
            if dropped_cols:
                st.success(f"Done! Dropped {len(dropped_cols)} columns: {', '.join(dropped_cols)}")
            else:
                st.info("No columns exceeded the threshold. Nothing was dropped.")

        # Save the cleaned data back to session state
        save_dataframe(df)
        st.rerun()  # reload the page to show the updated data

# Tab 2 — Remove duplicates
with tab2:
    dupe_count = int(df.duplicated().sum())
    st.metric("Current duplicate rows", dupe_count)

    if dupe_count > 0:
        if st.button("🗑️ Remove All Duplicate Rows", use_container_width=True):
            df, removed = remove_duplicates(df)
            save_dataframe(df)
            st.success(f"Done! Removed {removed} duplicate rows.")
            st.rerun()
    else:
        st.success("Your data has no duplicates. Nothing to do!")

# Tab 3 — Drop columns
with tab3:
    st.write("Select columns you want to permanently remove from the dataset:")

    # st.multiselect lets the user pick multiple items from a list
    columns_to_remove = st.multiselect(
        "Select columns to drop:",
        options=df.columns.tolist()
    )

    if st.button("🗑️ Drop Selected Columns", use_container_width=True):
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
            save_dataframe(df)
            st.success(f"Done! Dropped: {', '.join(columns_to_remove)}")
            st.rerun()
        else:
            st.warning("Please select at least one column first.")

# Tab 4 — Fix data types
with tab4:
    st.write(
        "Some columns that contain numbers might be stored as text. "
        "This button converts them to proper number format."
    )
    if st.button("🔧 Auto-Fix Data Types", use_container_width=True):
        df, converted = auto_fix_types(df)
        save_dataframe(df)
        if converted:
            st.success(f"Converted to numeric: {', '.join(converted)}")
        else:
            st.info("All columns already have the correct type. Nothing changed.")
        st.rerun()

# Reset button - restores the original uploaded data
st.markdown("---")
if st.button("↩️ Reset — Undo All Cleaning", use_container_width=True):
    st.session_state["df"] = st.session_state["df_original"].copy()
    st.success("Data has been reset to the original uploaded version.")
    st.rerun()

# ── Step 5: Charts ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 5 — Auto-Generated Charts")

# get_auto_charts returns a dictionary of {chart_name: chart_object}
charts = get_auto_charts(df)

if charts:
    # Create one tab per chart
    chart_tabs = st.tabs(list(charts.keys()))
    for i, (tab, (chart_name, chart)) in enumerate(zip(chart_tabs, charts.items())):
        with tab:
            # key=f"upload_chart_{i}" gives each chart a unique ID
            # Without this, Streamlit crashes when two charts look similar
            st.plotly_chart(chart, use_container_width=True, key=f"upload_chart_{i}")
else:
    st.info("No charts available. Your dataset may not have numeric columns.")

# ── Step 6: AI Insights ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 6 — AI Business Insights")
st.write("Click the button to get 5 AI-generated insights about your data.")

if st.button("✨ Generate AI Insights", use_container_width=True):
    # Show a loading spinner while we wait
    with st.spinner("Analysing your data, please wait..."):
        insights_text, source = generate_insights(df)

    # Show where the insights came from
    if source == "groq":
        st.caption("🦙 Powered by Llama 3.3 via Groq (free AI)")
    else:
        st.caption("📊 Using statistical analysis (set GROQ_API_KEY for AI insights)")

    # Display each insight as a separate blue info box
    for line in insights_text.strip().split("\n"):
        if line.strip():  # skip empty lines
            st.info(line.strip())

# ── Step 7: Anomaly Detection ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 7 — Anomaly Detection")
st.write("Find unusual rows in your data using 3 different methods.")

if st.button("🔍 Find Anomalies", use_container_width=True):
    with st.spinner("Scanning for anomalies..."):
        results = detect_anomalies(df)

    if "error" in results:
        st.error(results["error"])
    else:
        # Show summary numbers
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Anomalous Rows Found",   results["isolation_forest_count"])
        col_b.metric("% of Total Data",        f"{results['isolation_forest_pct']}%")
        col_c.metric("Columns with Z-Outliers", len(results["zscore_outliers"]))

        # Show the actual anomalous rows
        if not results["anomaly_rows"].empty:
            st.write("**Sample anomalous rows (Isolation Forest method):**")
            st.dataframe(results["anomaly_rows"], use_container_width=True)

        # Z-Score results
        if results["zscore_outliers"]:
            st.write("**Outliers found by Z-Score method:**")
            for col, count in results["zscore_outliers"].items():
                st.write(f"• **{col}**: {count} outliers (values more than 3 std deviations from mean)")

        # IQR results
        if results["iqr_outliers"]:
            st.write("**Outliers found by IQR method:**")
            for col, info in results["iqr_outliers"].items():
                st.write(
                    f"• **{col}**: {info['count']} outliers "
                    f"(normal range: {info['lower_bound']} to {info['upper_bound']})"
                )