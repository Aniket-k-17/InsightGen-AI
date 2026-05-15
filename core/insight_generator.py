# core/insight_generator.py
# This file generates AI business insights about the uploaded data.
# It tries to use Groq (free AI API). If that fails, it uses statistics instead.

import os          # os lets us read environment variables (like API keys)
import requests    # requests lets us call web APIs (like sending a web form)
from dotenv import load_dotenv  # reads the .env file

# Load the .env file so os.getenv() can read GROQ_API_KEY
load_dotenv()

# Read the API key from the .env file
# os.getenv() returns "" (empty string) if the key is not set
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq API details - these never change
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # the free Llama 3 model on Groq


def call_groq_api(system_message, user_message):
    """
    Calls the Groq API and returns the AI's response as text.
    Returns None if the API call fails for any reason.

    system_message = instructions telling the AI what role to play
    user_message   = the actual question / data we send
    """

    # If there is no API key, don't even try to call the API
    if not GROQ_API_KEY:
        return None

    # Headers are like metadata sent with the request
    # Authorization tells the API "here is my key, I am allowed to use this"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    # The data we send to the API (like a form submission)
    data = {
        "model":       GROQ_MODEL,
        "max_tokens":  600,   # maximum length of the AI's response
        "temperature": 0.7,   # creativity level (0=robotic, 1=very creative)
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
    }

    try:
        # requests.post() sends data to the API and waits for a response
        # timeout=30 means give up if no response after 30 seconds
        response = requests.post(GROQ_URL, headers=headers, json=data, timeout=30)

        # status_code 200 means success
        if response.status_code == 200:
            # The API returns JSON - we navigate to the text part
            ai_text = response.json()["choices"][0]["message"]["content"]
            return ai_text.strip()
        else:
            return None  # something went wrong with the API

    except Exception:
        return None  # network error or timeout - just return None


def generate_insights(df):
    """
    Main function - generates 5 business insights about the DataFrame.
    Returns: (insights_text, source)
    source = "groq" if AI was used, "statistics" if fallback was used
    """

    # Build a text summary of the dataset to send to the AI
    column_names = ", ".join(df.columns.tolist())
    shape_info   = f"{df.shape[0]} rows and {df.shape[1]} columns"
    stats_text   = df.describe().round(2).to_string()
    sample_rows  = df.head(5).to_csv(index=False)

    # Tell the AI what role to play
    system_message = (
        "You are a senior business data analyst. "
        "Generate exactly 5 clear, specific, actionable business insights. "
        "Format as: 1. insight  2. insight  etc. "
        "Always mention column names and actual numbers from the data."
    )

    # The actual data we want insights about
    user_message = (
        f"Dataset size: {shape_info}\n"
        f"Column names: {column_names}\n\n"
        f"Statistics:\n{stats_text}\n\n"
        f"First 5 rows:\n{sample_rows}"
    )

    # Try the Groq API first
    ai_response = call_groq_api(system_message, user_message)

    if ai_response:
        return ai_response, "groq"
    else:
        # API failed or no key - use statistics instead
        return generate_statistical_insights(df), "statistics"


def generate_statistical_insights(df):
    """
    Generates insights using only pandas statistics.
    No internet or API key needed - always works.
    """
    insights = []

    # Get lists of column types
    number_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    text_columns   = df.select_dtypes(include=["object"]).columns.tolist()

    # Insight 1 - dataset size
    insights.append(
        f"1. This dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns** "
        f"({len(number_columns)} numeric, {len(text_columns)} categorical)."
    )

    # Insight 2 - missing values
    total_missing = int(df.isnull().sum().sum())
    if total_missing == 0:
        insights.append("2. ✅ No missing values found. Data quality is excellent.")
    else:
        worst_column = df.isnull().sum().idxmax()  # column with most missing
        worst_pct    = round(df[worst_column].isnull().mean() * 100, 1)
        insights.append(
            f"2. ⚠️ **{total_missing} missing values** found. "
            f"Column `{worst_column}` has the most gaps ({worst_pct}% missing)."
        )

    # Insight 3 - most variable column
    if number_columns:
        most_variable = df[number_columns].std().idxmax()  # column with highest std dev
        col_mean = round(df[most_variable].mean(), 2)
        col_std  = round(df[most_variable].std(), 2)
        insights.append(
            f"3. `{most_variable}` has the highest variability "
            f"(average = {col_mean}, spread = {col_std}). "
            "This column has the most inconsistency in the data."
        )

    # Insight 4 - strongest correlation
    if len(number_columns) >= 2:
        corr_matrix = df[number_columns].corr()

        # Find the highest correlation pair (not comparing a column with itself)
        best_value = 0
        best_col1  = ""
        best_col2  = ""

        for i in range(len(number_columns)):
            for j in range(i + 1, len(number_columns)):  # avoid repeats
                value = corr_matrix.iloc[i, j]
                if abs(value) > abs(best_value):
                    best_value = value
                    best_col1  = number_columns[i]
                    best_col2  = number_columns[j]

        direction = "positively" if best_value > 0 else "negatively"
        insights.append(
            f"4. `{best_col1}` and `{best_col2}` are **{direction} correlated** "
            f"(r = {round(best_value, 2)}). They tend to move together."
        )

    # Insight 5 - most common category or value range
    if text_columns:
        col       = text_columns[0]
        top_value = df[col].value_counts().index[0]
        top_pct   = round(df[col].value_counts().iloc[0] / len(df) * 100, 1)
        insights.append(
            f"5. In `{col}`, the most common value is **'{top_value}'** "
            f"({top_pct}% of all rows)."
        )
    elif number_columns:
        col = number_columns[0]
        insights.append(
            f"5. `{col}` ranges from **{round(df[col].min(), 2)}** "
            f"to **{round(df[col].max(), 2)}** "
            f"with a median of **{round(df[col].median(), 2)}**."
        )

    # Join all insights with a blank line between each
    return "\n\n".join(insights)
