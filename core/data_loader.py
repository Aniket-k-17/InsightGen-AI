# core/data_loader.py
# This file handles LOADING files (CSV, Excel, JSON) into a pandas DataFrame.
# A DataFrame is like a table with rows and columns - think of it like Excel in Python.

import pandas as pd  # pandas is the main library for working with data tables


def load_file(uploaded_file):
    """
    Takes a file uploaded through Streamlit and returns a pandas DataFrame.
    Supports: CSV, Excel (.xlsx, .xls), JSON
    """

    # Get the file extension (e.g. "csv", "xlsx", "json")
    # .name gives the filename, .split(".") splits by dot, [-1] gets the last part
    file_extension = uploaded_file.name.split(".")[-1].lower()

    # Handle CSV files
    if file_extension == "csv":
        # Sometimes CSV files use different text encodings
        # We try 3 common ones until one works
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                uploaded_file.seek(0)  # go back to start of file before reading
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df  # return as soon as we successfully read it
            except UnicodeDecodeError:
                continue  # try the next encoding if this one failed

        # If all encodings failed, raise an error
        raise ValueError("Could not read CSV file. Try saving it as UTF-8.")

    # Handle Excel files
    elif file_extension in ["xlsx", "xls"]:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        return df

    # Handle JSON files
    elif file_extension == "json":
        uploaded_file.seek(0)
        df = pd.read_json(uploaded_file)
        return df

    # File type not supported
    else:
        raise ValueError(
            f"File type '.{file_extension}' is not supported. "
            "Please upload a CSV, Excel, or JSON file."
        )


def get_basic_info(df):
    """
    Takes a DataFrame and returns a dictionary with basic facts about it.
    A dictionary is a collection of key-value pairs, like: {"rows": 100, "columns": 5}
    """

    # select_dtypes picks columns by their data type
    # include="number" gets columns with numbers (int, float)
    # include="object" gets columns with text
    numeric_columns     = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    return {
        "rows":        df.shape[0],       # number of rows
        "columns":     df.shape[1],       # number of columns
        "numeric":     numeric_columns,   # list of number column names
        "categorical": categorical_columns, # list of text column names
        "missing":     int(df.isnull().sum().sum()),  # total missing values
        "duplicates":  int(df.duplicated().sum()),    # total duplicate rows
    }
