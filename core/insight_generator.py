# core/insight_generator.py
# Generates AI business insights using Groq's free Llama 3 model.
# Works locally (.env) and on Streamlit Cloud (st.secrets).

import requests
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def generate_insights(df):
    """
    Returns (insights_text, source) tuple.
    source = "groq" if AI was used, "local" if fallback statistics used.
    This is what upload.py expects.
    """
    api_key = _get_api_key()

    if not api_key:
        return _local_insights(df), "local"

    columns     = ", ".join(df.columns.tolist())
    shape       = f"{df.shape[0]} rows and {df.shape[1]} columns"
    stats       = df.describe().round(2).to_string()
    sample_data = df.head(5).to_csv(index=False)

    prompt = f"""You are a senior business data analyst.

Here is a dataset summary:
- Size: {shape}
- Columns: {columns}

Statistical Summary:
{stats}

Sample Data:
{sample_data}

Generate exactly 5 business insights from this dataset.
Format your response as:
1. insight here
2. insight here
3. insight here
4. insight here
5. insight here

Be specific, mention column names and numbers."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json"
    }

    payload = {
        "model":       "llama-3.3-70b-versatile",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  500,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"].strip()
            return text, "groq"
        return _local_insights(df), "local"

    except Exception:
        return _local_insights(df), "local"


# Keep old name as alias so nothing else breaks
def generate_ai_insights(df):
    text, _ = generate_insights(df)
    return text


def _local_insights(df):
    """
    Pure statistics fallback — no API needed.
    """
    insights = []
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    insights.append(
        f"1. The dataset has {df.shape[0]:,} rows and {df.shape[1]} columns "
        f"with {len(num_cols)} numeric and {len(cat_cols)} categorical features."
    )

    missing = df.isnull().sum().sum()
    if missing == 0:
        insights.append("2. Data quality is excellent — no missing values found across all columns.")
    else:
        worst = df.isnull().sum().idxmax()
        insights.append(
            f"2. Data has {missing} missing values total. "
            f"Column '{worst}' has the most — consider filling or dropping it."
        )

    if num_cols:
        top = df[num_cols].std().idxmax()
        insights.append(
            f"3. '{top}' shows the highest variability "
            f"(mean={round(df[top].mean(),2)}, std={round(df[top].std(),2)}). "
            f"This column likely drives the most change in the dataset."
        )

    if len(num_cols) >= 2:
        corr  = df[num_cols].corr()
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                pairs.append((abs(corr.iloc[i,j]), num_cols[i], num_cols[j], corr.iloc[i,j]))
        if pairs:
            pairs.sort(reverse=True)
            _, c1, c2, raw = pairs[0]
            direction = "positively" if raw > 0 else "negatively"
            insights.append(
                f"4. '{c1}' and '{c2}' are strongly {direction} correlated ({round(raw,2)}). "
                f"They move together — useful for prediction models."
            )

    if cat_cols:
        cat = cat_cols[0]
        top = df[cat].value_counts().index[0]
        pct = round(df[cat].value_counts().iloc[0] / len(df) * 100, 1)
        insights.append(
            f"5. In '{cat}', the most common value is '{top}' "
            f"appearing in {pct}% of records."
        )
    elif num_cols:
        col = num_cols[0]
        insights.append(
            f"5. '{col}' ranges from {round(df[col].min(),2)} to {round(df[col].max(),2)} "
            f"with a median of {round(df[col].median(),2)}."
        )

    return "\n\n".join(insights)