# pages/predict.py
# ML Prediction page — train a model and make live predictions.

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Fix blank charts on Streamlit Cloud
pio.renderers.default = "svg"

from models.ml_model import get_problem_type, train_model, predict_single
from utils.helpers   import stop_if_no_data, get_filename

stop_if_no_data()

df       = st.session_state["df"]
filename = get_filename()
all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

st.title("🧠 ML Prediction")
st.caption(f"Dataset: **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
st.markdown("---")

# ── Step 1: Target ────────────────────────────────────────────────────────────
st.subheader("Step 1 — What do you want to predict?")
target_col   = st.selectbox("🎯 Target column:", all_cols)
problem_type = get_problem_type(df, target_col)

if problem_type == "regression":
    st.success("📈 **Regression** — model will predict a number")
else:
    st.info(f"🏷️ **Classification** — model will predict a category ({df[target_col].nunique()} classes)")

# ── Step 2: Features ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 2 — Which columns to use as inputs?")

feature_options  = [c for c in all_cols if c != target_col]
default_features = [c for c in num_cols if c != target_col][:6]

feature_cols = st.multiselect(
    "📊 Feature columns (text columns are encoded automatically):",
    options=feature_options,
    default=default_features,
)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

text_features = [c for c in feature_cols if df[c].dtype == "object"]
if text_features:
    st.info(f"🔤 Text columns detected: **{', '.join(text_features)}** — auto-converted to numbers.")

# ── Step 3: Model ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 3 — Choose a model")

if problem_type == "regression":
    model_choice = st.radio("Model:", ["Linear Regression", "Decision Tree", "Random Forest"])
else:
    model_choice = st.radio("Model:", ["Logistic Regression", "Decision Tree", "Random Forest"])

descriptions = {
    "Linear Regression":   "Draws a straight line. Simple and fast.",
    "Logistic Regression": "Predicts probability of each category.",
    "Decision Tree":       "Asks yes/no questions to reach a prediction.",
    "Random Forest":       "Combines 100 decision trees. Usually most accurate.",
}
st.caption(f"ℹ️ {descriptions[model_choice]}")

# ── Step 4: Train ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 4 — Train the model")

test_size = st.slider("Test data %", 10, 40, 20, 5)
st.caption(f"Model trains on {100-test_size}% of rows, tested on remaining {test_size}%")

if st.button("🚀 Train Model", use_container_width=True):
    with st.spinner("Training... this may take a few seconds."):
        try:
            result = train_model(df, feature_cols, target_col, model_choice, test_size/100)
            st.session_state["ml_result"]   = result
            st.session_state["ml_features"] = feature_cols
            st.session_state["ml_target"]   = target_col
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.stop()

if "ml_result" not in st.session_state:
    st.info("👆 Click **Train Model** to see results.")
    st.stop()

result       = st.session_state["ml_result"]
feature_cols = st.session_state["ml_features"]
target_col   = st.session_state["ml_target"]

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Results")

c1, c2, c3 = st.columns(3)
c1.metric("Model",      result["model_name"])
c2.metric("Score",      f"{result['score']}%",
          help="R² for regression. Accuracy % for classification.")
c3.metric("Train rows", result["train_rows"])

# Only show MAE for regression — error is None for classification
if result["error"] is not None:
    st.metric("Avg Error (MAE)", result["error"])

score = result["score"]
if score >= 85:
    st.success(f"🎉 Great result! Score = {score}%")
elif score >= 65:
    st.warning(f"👍 Decent result. Score = {score}% — try Random Forest or add more features.")
else:
    st.error(f"⚠️ Weak result. Score = {score}% — try Random Forest or add more feature columns.")

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2 = st.tabs(["📈 Predictions Chart", "🏆 Feature Importance"])

def _dark():
    return dict(
        paper_bgcolor="#161b25", plot_bgcolor="#0d0f14",
        font=dict(color="#c8d0e0"), margin=dict(l=40,r=20,t=50,b=40)
    )

with tab1:
    if result["problem_type"] == "regression":
        chart_df = pd.DataFrame({"Actual": result["y_test"], "Predicted": result["y_pred"]})
        fig = px.scatter(chart_df, x="Actual", y="Predicted",
                         title="Actual vs Predicted — dots near green line = good model",
                         opacity=0.6, color_discrete_sequence=["#4f8ef7"])
        mn = min(chart_df["Actual"].min(), chart_df["Predicted"].min())
        mx = max(chart_df["Actual"].max(), chart_df["Predicted"].max())
        fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(color="#2dd4a0", dash="dash", width=2))
        fig.update_layout(**_dark())
        fig.update_xaxes(gridcolor="#1e2535")
        fig.update_yaxes(gridcolor="#1e2535")
        st.plotly_chart(fig, use_container_width=True, key="pred_chart")
    else:
        actual    = result["y_test"]
        predicted = result["y_pred"]
        correct   = sum(a == p for a, p in zip(actual, predicted))
        wrong     = len(actual) - correct
        count_df  = pd.DataFrame({
            "Result": ["✅ Correct", "❌ Wrong"],
            "Count":  [correct, wrong],
        })
        fig = px.bar(count_df, x="Result", y="Count",
                     title=f"Predictions — {correct}/{len(actual)} correct",
                     color="Result",
                     color_discrete_map={"✅ Correct": "#2dd4a0", "❌ Wrong": "#f77777"})
        fig.update_layout(**_dark())
        st.plotly_chart(fig, use_container_width=True, key="pred_chart")

with tab2:
    if result["importance"] is not None:
        fig = px.bar(result["importance"], x="Importance", y="Feature",
                     orientation="h",
                     title="Feature Importance — which columns matter most?",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(**_dark())
        st.plotly_chart(fig, use_container_width=True, key="importance_chart")
    else:
        st.info("Feature importance only available for Decision Tree and Random Forest.")

# ── Sample predictions table ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Sample Predictions (first 15 rows)")

actual_15    = result["y_test"][:15]
predicted_15 = result["y_pred"][:15]

# round() crashes on strings — only round for regression
if result["problem_type"] == "regression":
    predicted_15 = [round(float(p), 3) for p in predicted_15]
    actual_15    = [round(float(a), 3) for a in actual_15]
else:
    predicted_15 = [str(p) for p in predicted_15]
    actual_15    = [str(a) for a in actual_15]

st.dataframe(
    pd.DataFrame({
        f"Actual {target_col}":    actual_15,
        f"Predicted {target_col}": predicted_15,
    }),
    use_container_width=True, hide_index=True
)

# ── Live Prediction ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Try a Live Prediction")
st.write(f"Fill in values — the model will predict **{target_col}**:")

user_inputs = {}
cols_ui     = st.columns(3)

for i, col in enumerate(feature_cols):
    with cols_ui[i % 3]:
        if df[col].dtype == "object":
            options = sorted(df[col].dropna().unique().tolist())
            user_inputs[col] = st.selectbox(col, options, key=f"inp_{col}")
        else:
            user_inputs[col] = st.number_input(
                col, value=round(float(df[col].mean()), 4), key=f"inp_{col}"
            )

if st.button("🔮 Predict", use_container_width=True):
    try:
        prediction = predict_single(result, feature_cols, user_inputs)
        if result["problem_type"] == "regression":
            st.success(f"### 🎯 Predicted **{target_col}** = **{round(float(prediction), 4)}**")
        else:
            st.success(f"### 🎯 Predicted **{target_col}** = **{prediction}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")