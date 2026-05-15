# utils/helpers.py
# Small reusable helper functions used across multiple pages.
# Instead of writing the same code in every page, we write it once here
# and import it wherever we need it.

import streamlit as st
import pandas as pd


def get_dataframe():
    """
    Returns the current DataFrame from Streamlit's session state.
    Session state = Streamlit's way of remembering data between page changes.
    Returns None if no file has been uploaded yet.
    """
    return st.session_state.get("df", None)


def get_filename():
    """Returns the uploaded file's name. Defaults to 'dataset' if none uploaded."""
    return st.session_state.get("filename", "dataset")


def save_dataframe(df):
    """Saves a modified DataFrame back to session state."""
    st.session_state["df"] = df


def stop_if_no_data():
    """
    Call this at the top of any page that needs data.
    If no data is uploaded, shows a warning and stops the page from loading further.
    """
    if "df" not in st.session_state:
        st.warning("⚠️ Please go to **📂 Upload** in the sidebar and upload a dataset first.")
        st.stop()  # st.stop() halts everything below this line


def get_completeness(df):
    """
    Returns what % of the data is filled in (not missing).
    Formula: (non-missing cells / total cells) * 100
    """
    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())

    if total_cells == 0:
        return 100.0

    completeness = (1 - missing_cells / total_cells) * 100
    return round(completeness, 1)


def get_number_columns(df):
    """Returns a list of column names that contain numbers."""
    return df.select_dtypes(include="number").columns.tolist()


def get_text_columns(df):
    """Returns a list of column names that contain text."""
    return df.select_dtypes(include="object").columns.tolist()
