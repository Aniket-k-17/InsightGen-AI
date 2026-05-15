# core/data_cleaner.py
# This file has functions for CLEANING data.
# Each function takes a DataFrame, does one cleaning job, and returns:
#   - the cleaned DataFrame
#   - a number or list describing what was changed (so we can show a message)

import pandas as pd


def get_quality_report(df):
    """
    Looks at the DataFrame and returns a report about data quality.
    This is used on the Upload page to show missing values, duplicates, etc.
    """

    # isnull() returns True/False for each cell - True means it is empty
    # .sum() counts the Trues per column
    missing_per_column = df.isnull().sum()

    # Only keep columns that actually HAVE missing values (count > 0)
    missing_per_column = missing_per_column[missing_per_column > 0]

    # duplicated() returns True for rows that are exact copies of another row
    duplicate_count = int(df.duplicated().sum())

    # Calculate what % of the data is complete (not missing)
    total_cells = df.shape[0] * df.shape[1]  # rows × columns
    missing_cells = int(df.isnull().sum().sum())
    completeness = round((1 - missing_cells / total_cells) * 100, 1)

    return {
        "missing_per_column": missing_per_column,
        "duplicate_count":    duplicate_count,
        "completeness":       completeness,
    }


def remove_duplicates(df):
    """Remove all duplicate rows. Returns (cleaned_df, number_removed)."""
    rows_before = len(df)
    df = df.drop_duplicates()   # removes duplicate rows
    df = df.reset_index(drop=True)  # resets row numbers back to 0,1,2,3...
    rows_removed = rows_before - len(df)
    return df, rows_removed


def drop_rows_with_missing(df):
    """Remove any row that has at least one empty cell."""
    rows_before = len(df)
    df = df.dropna()  # dropna() removes rows with any NaN (empty) value
    df = df.reset_index(drop=True)
    rows_removed = rows_before - len(df)
    return df, rows_removed


def fill_with_mean(df):
    """
    Fill empty cells in NUMBER columns with that column's AVERAGE.
    Mean = sum of all values / count of values
    """
    number_columns = df.select_dtypes(include="number").columns
    cells_filled = int(df[number_columns].isnull().sum().sum())
    df = df.copy()  # make a copy so we don't change the original
    df[number_columns] = df[number_columns].fillna(df[number_columns].mean())
    return df, cells_filled


def fill_with_median(df):
    """
    Fill empty cells in NUMBER columns with that column's MEDIAN.
    Median = the middle value when sorted. Better than mean for skewed data.
    """
    number_columns = df.select_dtypes(include="number").columns
    cells_filled = int(df[number_columns].isnull().sum().sum())
    df = df.copy()
    df[number_columns] = df[number_columns].fillna(df[number_columns].median())
    return df, cells_filled


def fill_with_mode(df):
    """
    Fill empty cells with that column's MODE (most common value).
    Works for both number and text columns.
    """
    cells_filled = int(df.isnull().sum().sum())
    df = df.copy()
    for column in df.columns:
        if df[column].isnull().any():  # only process columns that have missing values
            most_common = df[column].mode()  # mode() returns most frequent value
            if len(most_common) > 0:
                df[column] = df[column].fillna(most_common[0])
    return df, cells_filled


def fill_with_custom_value(df, value):
    """Fill ALL empty cells with a specific value the user types in."""
    cells_filled = int(df.isnull().sum().sum())
    df = df.fillna(value)
    return df, cells_filled


def drop_high_missing_columns(df, threshold):
    """
    Remove columns where more than X% of values are missing.
    threshold = 50 means: drop columns that are more than 50% empty
    """
    # Calculate missing % for each column
    missing_percent = df.isnull().mean() * 100

    # Find columns that exceed the threshold
    columns_to_drop = missing_percent[missing_percent > threshold].index.tolist()

    df = df.drop(columns=columns_to_drop)
    return df, columns_to_drop


def auto_fix_types(df):
    """
    Try to convert TEXT columns to NUMBER columns when possible.
    Example: a column with values "1", "2", "3" should be numbers not text.
    """
    df = df.copy()
    converted_columns = []

    for column in df.select_dtypes(include="object").columns:
        try:
            # pd.to_numeric() tries to convert a column to numbers
            df[column] = pd.to_numeric(df[column])
            converted_columns.append(column)
        except (ValueError, TypeError):
            pass  # if conversion fails, just skip this column

    return df, converted_columns
