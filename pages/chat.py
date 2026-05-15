# pages/chat.py
# The CHAT page lets users have a conversation with an AI assistant.
# The AI knows about the uploaded dataset and can answer questions about it.
# Powered by Groq (free) using the Llama 3.3 model.

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
from dotenv import load_dotenv
from utils.helpers import get_filename, get_number_columns, get_text_columns

# Load the .env file FIRST, before reading any environment variables
load_dotenv()

# Now it is safe to read the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq API settings
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🤖 AI Chat Assistant")
st.caption("Ask anything about your data, data science, Python, or analysis")
st.markdown("---")

# ── Set up message history ────────────────────────────────────────────────────
# session_state["messages"] is a list of all chat messages
# Each message is a dictionary: {"role": "user" or "assistant", "content": "text"}
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Get dataset info if available (so the AI knows about the user's data)
df       = st.session_state.get("df", None)
filename = get_filename()


def build_system_prompt():
    """
    Creates the instructions that tell the AI what role to play.
    If a dataset is loaded, we include info about it so the AI can answer
    questions about the user's specific data.
    """
    # Base instructions for all conversations
    base = (
        "You are InsightGen AI, a friendly data analyst assistant. "
        "Help the user understand data science concepts, "
        "explain their dataset, suggest Python code, and give business advice. "
        "Keep answers clear and concise. Use simple language."
    )

    # If a dataset has been uploaded, add it as context
    if df is not None:
        num_cols  = get_number_columns(df)
        text_cols = get_text_columns(df)
        dataset_info = (
            f"\n\nThe user has uploaded a file called '{filename}' "
            f"with {df.shape[0]:,} rows and {df.shape[1]} columns. "
            f"Numeric columns: {num_cols}. "
            f"Text columns: {text_cols}. "
            f"Missing values: {int(df.isnull().sum().sum())}. "
            "Use this when answering questions about their data."
        )
        return base + dataset_info

    return base


def send_message_to_groq(all_messages):
    """
    Sends the full conversation history to Groq and returns the AI's reply.
    We send the whole history so the AI remembers what was said earlier.
    """
    # If there is no API key, return a helpful setup message instead
    if not GROQ_API_KEY:
        return (
            "⚠️ **No API key found.**\n\n"
            "To enable the AI chat, follow these steps:\n\n"
            "1. Go to 👉 **https://console.groq.com**\n"
            "2. Create a free account (no credit card needed)\n"
            "3. Click **API Keys** → **Create API Key**\n"
            "4. Open the `.env` file in your project folder\n"
            "5. Replace `your_groq_api_key_here` with your key\n"
            "6. Save and restart the app\n\n"
            "✅ The Groq free tier gives you **14,400 requests per day**!"
        )

    # Build the list of messages to send to Groq
    # Always include the system message first, then the full conversation
    messages_to_send = [{"role": "system", "content": build_system_prompt()}]

    # Add every message from the chat history
    for msg in all_messages:
        if msg["role"] in ("user", "assistant"):  # only send user and AI messages
            messages_to_send.append({
                "role":    msg["role"],
                "content": msg["content"],
            })

    # Call the Groq API
    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       GROQ_MODEL,
                "max_tokens":  1024,   # max length of AI response
                "temperature": 0.7,    # creativity (0=robotic, 1=very creative)
                "messages":    messages_to_send,
            },
            timeout=30,  # give up after 30 seconds
        )

        if response.status_code == 200:
            # Navigate the JSON response to get the text
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            error_msg = response.json().get("error", {}).get("message", "Unknown error")
            return f"❌ API Error {response.status_code}: {error_msg}"

    except requests.exceptions.Timeout:
        return "⏱️ The request timed out. Please try again."
    except Exception as e:
        return f"❌ Something went wrong: {str(e)}"


# ── Status banner ─────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error(
        "🔑 **API key missing.** "
        "Get a free key at [console.groq.com](https://console.groq.com) "
        "and add it to your `.env` file."
    )
elif df is not None:
    st.success(f"✅ AI ready · Dataset loaded: **{filename}** ({df.shape[0]:,} rows)")
else:
    st.info("✅ AI ready · No dataset loaded. Upload one for data-specific answers.")

# ── Quick starter buttons ──────────────────────────────────────────────────────
# Show example questions only when the chat is empty
if not st.session_state["messages"]:
    st.write("**Not sure what to ask? Try one of these:**")

    # Show different starters depending on whether data is loaded
    if df is not None:
        starter_questions = [
            f"Summarise the dataset '{filename}' for me",
            "What are the most important columns in this data?",
            "What ML model would work best for this dataset?",
            "Are there any data quality issues I should fix?",
        ]
    else:
        starter_questions = [
            "What is the difference between mean and median?",
            "How do I handle missing data in pandas?",
            "What is machine learning? Explain simply.",
            "Write Python code to load a CSV file with pandas",
        ]

    # Show 2 buttons per row
    btn_col1, btn_col2 = st.columns(2)
    for i, question in enumerate(starter_questions):
        button_col = btn_col1 if i % 2 == 0 else btn_col2
        if button_col.button(question, use_container_width=True, key=f"starter_{i}"):
            # When clicked, add question to history and get AI reply
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                reply = send_message_to_groq(st.session_state["messages"])
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.rerun()

    st.markdown("---")

# ── Chat history display ──────────────────────────────────────────────────────
# Loop through every message and display it
for message in st.session_state["messages"]:
    # st.chat_message creates a chat bubble with the right icon (user or robot)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input box ────────────────────────────────────────────────────────────
# st.chat_input creates the text box at the bottom of the screen
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show the user's message immediately
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show the AI's reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = send_message_to_groq(st.session_state["messages"])
        st.markdown(reply)

    # Save the AI's reply to history
    st.session_state["messages"].append({"role": "assistant", "content": reply})

# ── Clear chat button ─────────────────────────────────────────────────────────
if st.session_state["messages"]:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()