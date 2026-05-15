# 📊 InsightGen AI

> Turn raw data into business insights instantly — 100% free.

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/AI-Groq%20Llama%203-f55036?logo=meta&logoColor=white)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-2dd4a0)](LICENSE)

**Live Demo:** 🔗 [insightgen-ai-by-aniket.streamlit.app](https://insightgen-ai-by-aniket.streamlit.app)

---

## 📸 Screenshots

| Home | Dashboard | ML Prediction |
|------|-----------|---------------|
| ![Home](https://via.placeholder.com/280x160/0d0f14/4f8ef7?text=Home+Page) | ![Dashboard](https://via.placeholder.com/280x160/0d0f14/2dd4a0?text=Dashboard) | ![ML](https://via.placeholder.com/280x160/0d0f14/7c5cfc?text=ML+Predict) |

---

## ✨ Features

| Page | What it does |
|------|-------------|
| 📂 **Upload & Clean** | CSV, Excel, JSON · Fix missing values · Remove duplicates · Fix data types |
| 📊 **Dashboard** | Auto charts · Correlation heatmap · Scatter · Bar · Custom chart builder |
| 🧠 **ML Prediction** | Regression & Classification · Random Forest, Decision Tree, Linear Regression · Live prediction |
| 🤖 **AI Chat** | Ask anything about your data · Powered by Llama 3.3 70B (free) |
| ✨ **AI Insights** | 5 business insights auto-generated from your dataset |
| 🚨 **Anomaly Detection** | Isolation Forest · Z-Score · IQR — 3 methods combined |
| 📄 **Reports** | Download cleaned data as CSV · Excel (multi-sheet) · TXT report |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/insightgen-ai.git
cd insightgen-ai

# 2. Create a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free Groq API key
# Get one at: https://console.groq.com (free, takes 2 min)
# Create a .env file in the project root:
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub (public repo)
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app** → select your repo
4. Set **Main file path** → `app.py`
5. Click **Advanced settings** → **Secrets** → paste:
   ```
   GROQ_API_KEY = "your_groq_key_here"
   ```
6. Click **Deploy** — live in ~2 minutes ✅

---

## 🛠️ Tech Stack

| Layer | Tech |
|-------|------|
| UI & Deployment | Streamlit |
| Data Processing | Pandas, NumPy |
| Charts | Plotly |
| Machine Learning | scikit-learn |
| AI / LLM | Groq API — Llama 3.3 70B (free) |
| Reports | OpenPyXL |

---

## 📁 Project Structure

```
insightgen-ai/
├── app.py                  # Home page (entry point)
├── requirements.txt        # Python dependencies
├── .env                    # Your API key (never commit this!)
├── .gitignore
├── .streamlit/
│   ├── config.toml         # Dark theme settings
│   └── pages.toml          # Sidebar navigation
├── pages/
│   ├── upload.py           # Upload & Clean page
│   ├── dashboard.py        # Dashboard page
│   ├── predict.py          # ML Prediction page
│   ├── chat.py             # AI Chat page
│   └── reports.py          # Reports page
├── core/
│   ├── data_loader.py      # File loading (CSV/Excel/JSON)
│   ├── data_cleaner.py     # Data cleaning logic
│   ├── visualization.py    # Chart generation
│   ├── insight_generator.py# AI insights via Groq
│   └── anomaly_detector.py # Anomaly detection
├── models/
│   └── ml_model.py         # ML training & prediction
└── utils/
    └── helpers.py          # Shared utilities
```

---

## 👨‍💻 Built by

**Aniket Kondhalkar**

Built as a full-stack data analytics portfolio project using Python and Streamlit.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](#)

---

## 📄 License

MIT License — free to use, modify, and share.
