# pages/predict.py
# ML Prediction page — train a model and make predictions on your data.

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from models.ml_model import get_problem_type, train_model, predict_single
from utils.helpers   import stop_if_no_data, get_filename

# Stop if no data uploaded
stop_if_no_data()

df       = st.session_state["df"]
filename = get_filename()
all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🧠 ML Prediction")
st.caption(f"Dataset: **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
st.markdown("---")

# ── Step 1: Pick target ───────────────────────────────────────────────────────
st.subheader("Step 1 — What do you want to predict?")

target_col = st.selectbox("🎯 Target column:", all_cols)

# Auto-detect and show problem type
problem_type = get_problem_type(df, target_col)
if problem_type == "regression":
    st.success(f"📈 **Regression** — model will predict a number")
else:
    st.info(f"🏷️ **Classification** — model will predict a category  ({df[target_col].nunique()} classes)")

# ── Step 2: Pick features ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 2 — Which columns to use as inputs?")

feature_options = [c for c in all_cols if c != target_col]
default_features = [c for c in num_cols if c != target_col][:6]

feature_cols = st.multiselect(
    "📊 Feature columns:",
    options=feature_options,
    default=default_features,
)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# ── Step 3: Pick model ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 3 — Choose a model")

if problem_type == "regression":
    model_choice = st.radio("Model:", ["Linear Regression", "Decision Tree", "Random Forest"])
else:
    model_choice = st.radio("Model:", ["Logistic Regression", "Decision Tree", "Random Forest"])

# Simple description of each model
descriptions = {
    "Linear Regression":   "Draws a straight line through your data. Simple and fast.",
    "Logistic Regression": "Predicts probability of each category. Good for yes/no problems.",
    "Decision Tree":       "Asks a series of yes/no questions to reach a prediction.",
    "Random Forest":       "Combines 100 decision trees. Usually the most accurate.",
}
st.caption(f"ℹ️ {descriptions[model_choice]}")

# ── Step 4: Train ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 4 — Train the model")

test_size = st.slider("Test data %", 10, 40, 20, 5)
st.caption(f"Model trains on {100 - test_size}% of rows, tested on {test_size}%")

if st.button("🚀 Train Model", use_container_width=True):
    with st.spinner("Training..."):
        try:
            result = train_model(df, feature_cols, target_col, model_choice, test_size / 100)
            st.session_state["ml_result"]      = result
            st.session_state["ml_features"]    = feature_cols
            st.session_state["ml_target"]      = target_col
            st.success("✅ Model trained!")
        except Exception as e:
            st.error(f"Training failed: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if "ml_result" not in st.session_state:
    st.info("👆 Click **Train Model** to see results.")
    st.stop()

result      = st.session_state["ml_result"]
feature_cols = st.session_state["ml_features"]
target_col  = st.session_state["ml_target"]

st.markdown("---")
st.subheader("📊 Results")

# Score metrics
c1, c2, c3 = st.columns(3)
c1.metric("Model",       result["model_name"] if "model_name" in result else model_choice)
c2.metric("Score",       f"{result['score']}%",
          help="R² for regression, Accuracy for classification")
c3.metric("Train rows",  result["train_rows"])

if result["error"]:
    st.metric("Avg Error (MAE)", result["error"],
              help="On average, predictions are off by this amount")

# Score feedback
score = result["score"]
if score >= 85:
    st.success(f"🎉 Great result! Score = {score}%")
elif score >= 65:
    st.warning(f"👍 Decent result. Score = {score}% — try a different model or more features.")
else:
    st.error(f"⚠️ Weak result. Score = {score}% — try Random Forest or add more feature columns.")

# Charts
st.markdown("---")
tab1, tab2 = st.tabs(["📈 Predictions Chart", "🏆 Feature Importance"])

with tab1:
    chart_df = pd.DataFrame({"Actual": result["y_test"], "Predicted": result["y_pred"]})

    if result["problem_type"] == "regression":
        fig = px.scatter(
            chart_df, x="Actual", y="Predicted",
            title="Actual vs Predicted (dots near the diagonal = good model)",
            template="plotly_dark", opacity=0.6,
        )
        # Draw perfect prediction line
        min_v = min(chart_df["Actual"].min(), chart_df["Predicted"].min())
        max_v = max(chart_df["Actual"].max(), chart_df["Predicted"].max())
        fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                      line=dict(color="#2dd4a0", dash="dash"))
    else:
        # For classification show a bar chart of prediction counts
        chart_df["Correct"] = chart_df["Actual"] == chart_df["Predicted"]
        count_df = chart_df["Correct"].value_counts().reset_index()
        count_df.columns = ["Result", "Count"]
        count_df["Result"] = count_df["Result"].map({True: "✅ Correct", False: "❌ Wrong"})
        fig = px.bar(count_df, x="Result", y="Count",
                     title="Correct vs Wrong Predictions",
                     color="Result", template="plotly_dark",
                     color_discrete_map={"✅ Correct": "#2dd4a0", "❌ Wrong": "#f77"})

    st.plotly_chart(fig, use_container_width=True, key="pred_chart")

with tab2:
    if result["importance"] is not None:
        fig = px.bar(
            result["importance"], x="Importance", y="Feature",
            orientation="h", title="Which columns influenced predictions most?",
            color="Importance", color_continuous_scale="Blues",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True, key="importance_chart")
    else:
        st.info("Feature importance not available for this model type.")

# Sample predictions table
st.markdown("---")
st.subheader("🔍 Sample Predictions (first 15 rows)")
sample = pd.DataFrame({
    "Actual":    result["y_test"][:15],
    "Predicted": [round(p, 3) for p in result["y_pred"][:15]],
})
st.dataframe(sample, use_container_width=True, hide_index=True)

# ── Live Prediction ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Try a Live Prediction")
st.write(f"Enter values below and the model will predict **{target_col}**:")

user_inputs = {}
cols = st.columns(3)

for i, col in enumerate(feature_cols):
    with cols[i % 3]:
        if df[col].dtype == "object":
            user_inputs[col] = st.selectbox(col, df[col].dropna().unique().tolist(), key=f"input_{col}")
        else:
            user_inputs[col] = st.number_input(
                col,
                value=float(df[col].mean()),
                key=f"input_{col}"
            )

if st.button("🔮 Predict", use_container_width=True):
    try:
        prediction = predict_single(result, feature_cols, user_inputs)
        if result["problem_type"] == "regression":
            st.success(f"### Predicted **{target_col}** = {round(float(prediction), 4)}")
        else:
            st.success(f"### Predicted **{target_col}** = {prediction}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")