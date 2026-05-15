# app.py
# Home page of InsightGen AI
# This is the first file Streamlit runs: streamlit run app.py

import streamlit as st

st.set_page_config(
    page_title="InsightGen AI",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 50%, #1a2744 100%);
    border: 1px solid #2a3a5c;
    border-radius: 16px;
    padding: 50px 40px;
    text-align: center;
    margin-bottom: 30px;
}
.hero h1 {
    font-size: 3em;
    font-weight: 800;
    background: linear-gradient(90deg, #4f8ef7, #7c5cfc, #2dd4a0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.hero p { font-size: 1.2em; color: #8b9cb5; margin: 0; }

/* Stats boxes */
.stat-box {
    background: #161b25;
    border: 1px solid #2a3a5c;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.stat-number { font-size: 2em; font-weight: 800; color: #4f8ef7; }
.stat-label  { font-size: 0.85em; color: #8b9cb5; margin-top: 4px; }

/* Feature cards
   KEY FIX: removed fixed height, added flexbox to centre content,
   added min-height so short-text cards still look tall enough        */
.card {
    background: #161b25;
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 22px 16px;
    text-align: center;
    min-height: 145px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: border-color 0.2s;
}
.card:hover  { border-color: #4f8ef7; }
.card-icon   { font-size: 2em; margin-bottom: 10px; line-height: 1; }
.card-title  { font-size: 1em; font-weight: 700; color: #e8eaf0; margin-bottom: 6px; }
.card-desc   { font-size: 0.82em; color: #8b9cb5; line-height: 1.5; }

/* How-to steps */
.step {
    background: #161b25;
    border-left: 4px solid #4f8ef7;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.step-num   { color: #4f8ef7; font-weight: 700; font-size: 0.85em; }
.step-title { color: #e8eaf0; font-weight: 600; margin: 3px 0; }
.step-desc  { color: #8b9cb5; font-size: 0.85em; line-height: 1.5; }

/* Footer */
.footer {
    text-align: center;
    color: #4a5568;
    font-size: 0.8em;
    padding: 20px 0 10px;
    border-top: 1px solid #2a3a5c;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="font-size:2.5em;">📊</div>
    <h1>InsightGen AI</h1>
    <p>Upload your data → Clean it → Visualise → Get AI Insights → Train ML Models</p>
    <p style="margin-top:12px; font-size:0.95em; color:#4f8ef7;">
        100% Free &nbsp;·&nbsp; No Credit Card &nbsp;·&nbsp; Powered by Llama 3
    </p>
</div>
""", unsafe_allow_html=True)


# ── Stats Bar ─────────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
stats = [
    ("3",    "File Formats Supported"),
    ("7",    "Chart Types"),
    ("6",    "ML Models Available"),
    ("100%", "Free to Use"),
]
for col, (number, label) in zip([s1, s2, s3, s4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{number}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Feature Cards ─────────────────────────────────────────────────────────────
st.markdown("### 🚀 What you can do")

# All 8 cards defined as a list — easy to read and edit
cards = [
    ("📂", "Upload & Clean",      "CSV, Excel, JSON · Fix missing values · Remove duplicates"),
    ("📊", "Dashboard",           "Auto charts · Correlation heatmap · Custom chart builder"),
    ("🧠", "ML Prediction",       "Regression & Classification · 3 models · Live prediction"),
    ("🤖", "AI Chat",             "Ask questions about your data · Powered by Llama 3 (free)"),
    ("✨", "AI Insights",         "5 business insights generated automatically from your data"),
    ("🚨", "Anomaly Detection",   "Isolation Forest · Z-Score · IQR — 3 methods combined"),
    ("📄", "Reports",             "Download as CSV · Excel (multi-sheet) · TXT report"),
    ("🧹", "Data Cleaning",       "Fill missing · Drop duplicates · Fix types · Reset anytime"),
]

# Row 1 — first 4 cards
cols_row1 = st.columns(4)
for col, (icon, title, desc) in zip(cols_row1, cards[:4]):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <div class="card-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 2 — last 4 cards
cols_row2 = st.columns(4)
for col, (icon, title, desc) in zip(cols_row2, cards[4:]):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <div class="card-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── How to use — full width now (API section removed) ─────────────────────────
st.markdown("### 📋 How to use")

steps = [
    ("STEP 1", "📂 Upload your file",        "Go to Upload in the sidebar. Drag and drop a CSV, Excel, or JSON file."),
    ("STEP 2", "🧹 Clean your data",          "Fix missing values, remove duplicates, and drop unwanted columns."),
    ("STEP 3", "📊 Explore the Dashboard",    "View auto-generated charts and build your own custom charts."),
    ("STEP 4", "🧠 Train an ML model",        "Pick a target column, choose a model, and make live predictions."),
    ("STEP 5", "📄 Download your report",     "Export your cleaned data and full analysis as CSV, Excel, or TXT."),
]

# Show steps in 2 columns so they don't stretch too wide
left, right = st.columns(2)
for i, (num, title, desc) in enumerate(steps):
    # First 3 steps on the left, last 2 on the right
    target_col = left if i < 3 else right
    with target_col:
        st.markdown(f"""
        <div class="step">
            <div class="step-num">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by Aniket Kondhalkar &nbsp;·&nbsp;
    Streamlit &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; Groq (Llama 3)
</div>
""", unsafe_allow_html=True)